import cv2
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from typing import Dict
import os
import time
from threading import Thread
from typing import List, Callable
import open3d as o3d

from app.app import App
from app.gpose_chooser import GraspPoseChooser
from lib.geometry import invert_homogeneous
from lib.utils import time_stamp
from app.logger import Logger
from models.cameras import LogPlayer, CamFrame, Camera, L515, D415
from models.cameras.realsense import WrongCameraException

from models.obstruction_pcd import ObstructionPcd
from networks.pose_detection_network import PoseDetectionNetwork

from models import DTObject, OnRobot_RG2
from models.robots.fanuc_rg2_D415 import Fanuc_RG2_D415
from tasks import task_list
from tasks.base_task import Task


class DigitalTwin:
    EYE_IN_HAND = "eye_in_hand"  # reserved name for cam on robot gripper
    LOG_PLAYER = "logplayer"

    BASE_LOGDIR = 'logs'

    camera_list: Dict[str, Camera]
    selected_camera: str

    dt_objectcs_list: Dict[str, DTObject]
    selected_dt_object: str

    def __init__(self, cb_camera: Callable[[CamFrame], None] = None):
        self.tasks: List[Task] = [x() for x in task_list]
        print("Available Tasks:\n\t", "\n\t".join(
            [x.name for x in self.tasks]))

        self.logger = Logger()
        self.gpose = GraspPoseChooser()

        # flags
        self.dirty = True  # marks if lists have changed
        self.stop_task = True
        self.stop_flag = False

        # states
        self.task_running = None
        self.selected_dt_object = None
        self.selected_camera = None
        self.selected_log = None

        # switches (and defaults)
        self.log = False
        self.single_log = False
        self.play_log = False
        self.lock_robot = True
        self.draw_object = False
        self.freeze_object = False
        self.use_icp = False
        self.simulate_robot = False

        self.camera_list = {}
        self.dt_objectcs_list = {}
        self.log_list = os.listdir(self.BASE_LOGDIR)
        self.threads: List[Thread] = []

        self.charuco_board = None
        self.ar_dict = None

        self.cb_camera = cb_camera

        self.robot = Fanuc_RG2_D415()

        self.gripper = self.robot.gripper
        self.robot_arm = self.robot.endeffector
        cam = self.robot.eye_in_hand

        self.camera_list.update({self.EYE_IN_HAND: cam})

        self.logplayer = LogPlayer(self.BASE_LOGDIR, name=self.LOG_PLAYER)
        self.camera_list.update({self.LOG_PLAYER: self.logplayer})

        self.obstruction_pcd = ObstructionPcd()

        self.create_dtObject("cpsduck")
        self.create_dtObject("stapler")
        self.create_dtObject("chew_toy")
        self.create_dtObject("cpsglue")
        self.create_dtObject("pliers")

        # register app for each task
        for _task in self.tasks:
            _task.init(self)

        self.selected_dt_object = list(self.dt_objectcs_list.keys())[0]
        self.selected_camera = list(self.camera_list.keys())[0]
        self.selected_log = self.log_list[0]

        self.threads.append(Thread(target=self.camera_thread))
        self.threads.append(Thread(target=self.slow_update_task))

    def start(self):
        self.robot.start()
        for thread in self.threads:
            thread.start()

    def is_task_cancelled(self):
        return self.stop_task

    def cancel_task(self):
        """ also when task finished """
        if self.task_running is None:
            print(f"[{time_stamp()}] No task is running")
            return
        self.stop_task = True
        self.task_running = None
        self.gripper.stop()

    def create_dtObject(self, obj_id):
        n_objs = sum([x.startswith(obj_id)
                     for x in self.dt_objectcs_list.keys()])
        unique_obj_name = f"{obj_id}__{n_objs:02}"
        meshpath = os.path.join("data", "models", obj_id, f"{obj_id}.ply")

        new_obj = DTObject(name=unique_obj_name,
                           meshpath=meshpath, show_axes=True)
        self.dt_objectcs_list.update({unique_obj_name: new_obj})

    def slow_update_task(self):
        """ everything that needs to be updated all 5 sec """
        while not self.stop_flag:
            time.sleep(5)

            log_list = os.listdir(self.BASE_LOGDIR)
            if log_list != self.log_list:
                self.log_list = log_list
                self.dirty = True
                print("changed")

    def camera_thread(self):
        from networks.pvn.pvn3d.main_darknet import MainDarknet

        poseDetections = {}
        # in the future single object detector for all objects -> decision logic
        objectDetectors = {}
        # only consider object type, not unique id
        # for obj in [x.split('__')[0] for x in self.dt_objectcs_list.keys()]:
        #    if obj not in poseDetections.keys():
        #        print(
        #            f"[{time_stamp()}] Creating PoseDetection Network for object {obj}")
        #        n_points = 512 if obj == 'cpsduck' else 256
        #        try:
        #            poseDetections[obj] = PoseDetectionNetwork(obj, n_points)
        #        except Exception:
        #            poseDetections[obj] = None

        objectDetector = MainDarknet()
        obj_data = os.path.join("data", "weights", "darknet", "cps")
        objectDetector.cfg_file = os.path.join(obj_data, "yolo.cfg")
        objectDetector.data_file = os.path.join(obj_data, "obj.data")
        objectDetector.weights_path = os.path.join(
            obj_data, "yolo.weights")
        objectDetector.initial_trainer_and_model()

        prev_pose = None
        prev_omega = None
        omega = None
        v = None
        prev_v = None
        frame = None
        prior_obj_type = None
        prior_cam = None
        detected = False
        t_last_cb_camera = time.perf_counter()
        dt_obj = self.dt_objectcs_list[self.selected_dt_object]

        while not self.stop_flag:
            # Time/FPS management
            settings = App().settings['Object Detection']  # relevant settings
            dt = time.perf_counter() - t_last_cb_camera
            if dt < 1/settings['FPS']:
                # limit to 30 FPS
                time.sleep(1/settings['FPS'] - dt)
                dt = time.perf_counter() - t_last_cb_camera
            t_last_cb_camera = time.perf_counter()

            # selected objects, cameras, networks
            if prior_cam != self.selected_camera:
                cam = self.camera_list[self.selected_camera]
                if self.selected_camera != self.LOG_PLAYER:
                    self.logplayer.hide()
                else:
                    self.logplayer.show()
                    self.logplayer.read_logs(self.selected_log)
                prior_cam = self.selected_camera

            dt_obj_type = self.selected_dt_object.split('__')[0]
            if prior_obj_type != dt_obj_type:
                prior_obj_type = dt_obj_type
                dt_obj.hide()  # hide 'old' dt object
                dt_obj = self.dt_objectcs_list[self.selected_dt_object]
                # poseDetection: PoseDetectionNetwork = poseDetections[dt_obj_type]
                n_points = 512
                poseDetection = PoseDetectionNetwork(dt_obj_type, n_points)

            frame = cam.grab_frame()

            if frame is None:
                continue

            # Image (2D object detection/prediction)
            bbox = None
            confidence_threshold = settings['confidence']
            darknet_frame, detections, _ = objectDetector.image_detection(
                frame.rgb.copy(), confidence_threshold, None)
            # print(detections)
            detections = [x for x in detections if x[0] == dt_obj_type]
            if len(detections) > 0:
                bbox_xywh = detections[-1][2]
                confidence = detections[-1][1]
                bbox = objectDetector.yolo_bbox_2_original(
                    bbox_xywh, frame.rgb.shape[:2])

            if detected:
                # if previously detected calculate the bounding box from pvn prediction
                # project the bounding box of the object into the image
                affine_matrix = invert_homogeneous(
                    frame.extrinsic) @ dt_obj.pose
                pvn_bbox = dt_obj.bounding_box @ affine_matrix[:3, :3].T
                pvn_bbox += affine_matrix[:3, 3].T
                bbox_in_img = pvn_bbox[:, :3]
                bbox_in_img /= bbox_in_img[:, 2:3]
                bbox_in_img[:, 2] = 1.
                bbox_in_img = bbox_in_img @ frame.intrinsic.T

                # get image bounding box
                x_min, y_min = np.min(bbox_in_img[:, :2], axis=0)
                x_max, y_max = np.max(bbox_in_img[:, :2], axis=0)
                pvn_bbox = np.array([x_min, y_min, x_max, y_max])
                pvn_bbox[::2] = np.clip(pvn_bbox[::2], 0, frame.rgb.shape[1])
                pvn_bbox[1::2] = np.clip(pvn_bbox[1::2], 0, frame.rgb.shape[0])
                bbox = pvn_bbox

            # 6D pose estimation of selected object
            if not self.freeze_object:
                detected = False
                affine_matrix = None
                if bbox is not None and poseDetection is not None:
                    try:
                        detected, affine_matrix = poseDetection.inference(bbox, frame.rgb.copy(
                        ), frame.depth, frame.intrinsic, use_icp=self.use_icp)  # affine_matrix.shape = 3x4 !
                    except Exception as e:
                        print(f"[{time_stamp()}] pose detection failed with ", e)

                if detected:
                    current_pose = frame.extrinsic @ affine_matrix
                    if prev_pose is not None:
                        v = np.linalg.norm(
                            current_pose[:3, 3] - prev_pose[:3, 3]) / dt  # m / sec
                        omega = R.from_matrix(
                            current_pose[:3, :3]) * R.from_matrix(prev_pose[:3, :3].T)
                        omega = omega.magnitude() / dt  # rad/sec
                        #print(f"Object [v|omega]: {v:.4f}, {omega/np.pi*180:.2f}")

                        if prev_v is not None:
                            v_dot = (v - prev_v) / dt
                            omega_dot = (omega - prev_omega) / dt
                            #print(f"Object [v_dot|omega_dot]: {v_dot:.4f}, {omega_dot/np.pi*180:.2f}")

                        vel_limit = v > 0.12
                        omega_limit = omega > np.pi*0.8
                        if vel_limit or omega_limit:
                            detected = False
                            reason = "velocity" if vel_limit else "omega"
                            print(
                                f"[{time_stamp()}] Using darknet because of {reason}")

                if detected:
                    prev_pose = current_pose
                    if prev_pose is not None:
                        prev_omega = omega
                        prev_v = v
                    else:
                        prev_omega = None
                        prev_v = None
                    dt_obj.show()
                    dt_obj.transform(current_pose)

                else:
                    dt_obj.hide()
                    prev_pose = None
                    prev_omega = None
                    prev_v = None

            else:
                # if object is frozen, update affine matrix due to change of camera position
                affine_matrix = invert_homogeneous(
                    frame.extrinsic) @ dt_obj.pose

            # update digital twin obstruction
            self.obstruction_pcd.update_from_scene(
                frame, dt_obj, affine_matrix)

            if self.log or self.single_log:
                data = {'time': time.perf_counter(), 'rgb': frame.rgb,
                        'depth': frame.depth, 'extrinsic': frame.extrinsic, 'intrinsic': frame.intrinsic, 'cam2obj': affine_matrix}

            if self.log:
                self.logger.log(data, self.logplayer.cam.logdir)

            if self.single_log:
                self.logger.log(data, self.logplayer.cam.logdir)
                self.single_log = False

            # annotate
            annotated = frame.rgb.copy()
            if self.draw_object:
                if bbox is not None:
                    annotated = cv2.rectangle(annotated, (int(bbox[0]), int(
                        bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), thickness=4)
                if dt_obj.active:
                    annotated = poseDetection.draw_obj(
                        annotated, affine_matrix,  frame.intrinsic)

            cv2.putText(annotated, f"{1/dt:4.1f} FPS", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=3)

            anno_frame = CamFrame(rgb=annotated, depth=frame.depth.copy(
            ), intrinsic=frame.intrinsic, extrinsic=frame.extrinsic, rgb2=darknet_frame)

            if self.cb_camera is not None:
                self.cb_camera(anno_frame)

    def control_robot(self, control):
        self.gripper.set_target_force(
            self.gripper.target_force + control[6])
        self.gripper.set_target_width(
            self.gripper.target_width + control[7])

        if np.any(control[:6]):
            new_pose = np.eye(4)
            new_pose[:3, :3] = R.from_euler(
                'XYZ', control[3:6]).as_matrix()
            new_pose[:3, 3] = control[0:3]

            self.gripper.move_to_pose(
                new_pose, incremental=True, local_frame=True)

    def close(self):
        print("Quitting...")
        self.robot.close()
        self.stop_flag = True
        for thread in self.threads:
            thread.join()
        for cam in self.camera_list.values():
            cam.close()
        self.logger.close()
        self.gpose.close()
        return True
