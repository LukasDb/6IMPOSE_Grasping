from .base_task import Task, TaskType
from app.gpose_chooser import GraspPoseChooserConfig, GraspPoseChooserJob
from models import DTObject
from models.base_gripper import BaseGripper
from models.obstruction_pcd import ObstructionPcd
from models.cameras.base_camera import Camera
from models.targetmodel import TargetModel

import data.poses as poses
from lib.trajectory import get_trajectory_step_pose, generate_traj
from lib.utils import time_stamp
from lib.geometry import homogeneous_mat_from_RT, invert_homogeneous, rotation_between_vectors
from scipy.spatial.transform import Rotation as R
import time
import os
import numpy as np


class BinPicking(Task):
    name = "Bin Picking"
    description = "Picks object from bin and drops it back (Open-Loop)"
    type = TaskType.JOB

    grip_dist = 0.1  # from which distance the robot will pick open-loop
    tangent_weight = 0.2  # tangential weight for trajectory generation

    def on_close(self):
        self.gripper_vis.remove()
        self.robot_arm.stop()
        self.app.freeze_object = False

    def execute(self):
        app = self.app
        self.obj: DTObject = app.dt_objectcs_list[app.selected_dt_object]
        self.obstruction_pcd: ObstructionPcd = app.obstruction_pcd
        self.robot_arm: TargetModel = app.robot_arm
        self.gripper: BaseGripper = app.gripper

        home_pose = poses.start_bin_pick

        self.gripper.open_gripper()
        print(f"[{time_stamp()}] To home...")
        self.gripper.move_to_pose(home_pose)

        gripper_collision = os.path.join(
            "data", "models", "gripper_collision.ply")
        self.gripper_vis = gripper_vis = DTObject(
            gripper_collision, name="grasp_pose", wire_frame=True, show_axes=True)
        gripper_vis.dt_mesh.paint_uniform_color([0.2, 0.2, 1])  # paint blu

        config = GraspPoseChooserConfig(gripper_collision, self.grip_dist)
        app.gpose.set_config(config)

        self.gripper.wait_for_move()

        print(f"[{time_stamp()}] Home reached. Approaching object...")
        cam: Camera = app.camera_list[app.EYE_IN_HAND]
        above = homogeneous_mat_from_RT(
            self.gripper.pose[:3, :3], self.obj.position)
        above[2, 3] += 0.38
        self.gripper.move_to_pose(above)
        self.gripper.wait_for_move()

        # point camera towards object
        üobj_pos = self.obj.position
        # cam.point_at(obj_pos)
        time.sleep(1.0)

        ind = None
        t_pose_finding = time.perf_counter()

        # TODO some mean ?

        print(f"[{time_stamp()}] Finding grasping pose ....")
        local_pre_grasps = self.gripper.get_pre_grasp_poses(self.obj)
        grasp_offsets_from_closed = [self.gripper.get_grasp_offset_from_closed(
            gpose.width) for gpose in self.obj.gripping_poses]

        while ind is None:  # wait until one grasp has 10 votes
            if app.is_task_cancelled():
                self.on_close()
                return True

            if (time.perf_counter() - t_pose_finding) > 10.0:
                print(f"[{time_stamp()}] Could not find safe grasp pose.")
                self.on_close()
                return False

            if not self.obj.active:
                gripper_vis.remove()
                time.sleep(0.2)
                continue

            obj_pose = self.obj.pose
            gripper_pose = self.gripper.pose

            # get hand score
            obstruction_points = self.obstruction_pcd.get_points()

            # convert to global pre-grasp poses
            pre_grasps = [obj_pose @
                          pre_grasp for pre_grasp in local_pre_grasps]

            job = GraspPoseChooserJob(obj_pose, gripper_pose, grasp_offsets_from_closed,
                                      obstruction_points, pre_grasps)
            ind = self.app.gpose.get_gpose(job)

        best_grasp = pre_grasps[ind]
        self.app.freeze_object = True

        gripper_vis.transform(best_grasp)

        #pre_grasp_width = self.obj.gripping_poses[ind].width + 0.015
        # self.gripper.set_target_width(pre_grasp_width)

        print("Found grasp pose. Moving to grasp the object...")
        while True:
            if app.is_task_cancelled():
                self.on_close()
                return True

            gripper_pose = self.gripper.pose

            dist_2_target = np.linalg.norm(
                gripper_pose[:3, 3] - best_grasp[:3, 3])
            if dist_2_target < self.grip_dist:
                print(f"[{time_stamp()}] SWITCHING TO OPEN-LOOP")
                break

            trajectory = generate_traj(
                gripper_pose, best_grasp, tangent_weight=self.tangent_weight)

            next_traj = get_trajectory_step_pose(
                trajectory, gripper_pose, best_grasp)

            self.gripper.move_to_pose(next_traj)
            #gripper_tip2robot = invert_homogeneous(self.gripper.link_mat)
            #self.robot_arm.move_to_pose(next_traj @ gripper_tip2robot)

        # grasp with pre-grasp pose
        #self.robot_arm.move_to_pose(next_traj @ gripper_tip2robot)
        # self.robot_arm.wait_for_move(timeout=20)
        self.gripper.move_to_pose(best_grasp, linear=True)
        self.gripper.wait_for_move(timeout=20)

        print(f"[{time_stamp()}] Gripping object...")
        self.gripper.close_gripper()
        # a bit safer? TODO test on robot
        # self.gripper.set_target_width(
        #     self.obj.gripping_poses[ind].width - 0.01)
        while self.gripper.gripper_width > self.obj.gripping_poses[ind].width + 0.001:
            time.sleep(0.1)
        time.sleep(0.5)
        print(f"[{time_stamp()}] Return to home...")
        # time.sleep(3)
        gripper_vis.remove()

        # simulate gripping
        # pick predefined local gpose
        # object pose matches then gripper pose @ link_mat (with actual grasp width) @ local gpose
        local_gpose = self.obj.gripping_poses[ind].pose
        self.obj.attach(self.gripper, invert_homogeneous(local_gpose))

        pull_away = np.eye(4)
        pull_away[2, 3] = 0.2

        self.gripper.move_to_pose(pull_away @ best_grasp, linear=True)
        while not self.gripper.wait_for_move(timeout=1):
            self.gripper.move_to_pose(pull_away @ best_grasp)

        self.gripper.move_to_pose(home_pose)
        while not self.gripper.wait_for_move(timeout=1):
            self.gripper.move_to_pose(home_pose)
        time.sleep(0.2)
        self.gripper.open_gripper()
        time.sleep(1)
        self.app.freeze_object = False
        self.on_close()
        return True
