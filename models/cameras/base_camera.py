import cv2
import numpy as np
import open3d as o3d
from models.targetmodel import TargetModel
import os
from lib.utils import time_stamp
from lib.geometry import get_affine_matrix_from_6d_vector, homogeneous_mat_from_RT, invert_homogeneous, rotation_between_vectors
from lib.camera_calibration import read_chessboards, load_arrays, load_images, cv2_calibration, optimize_camera_offeset
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CamFrame():
    rgb: np.ndarray = None
    rgb2: np.ndarray = None
    depth: np.ndarray = None
    extrinsic: np.ndarray = None
    intrinsic: np.ndarray = None


class Camera(TargetModel, ABC):
    has_depth: bool
    has_stereo: bool
    cal_path: str
    img_path: str
    intrinsic_matrix: np.ndarray
    extrinsic_matrix: np.ndarray
    dist_coeffs: np.ndarray
    baseline: float
    hfov: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        """ to be called after initialization """
        pass

    @abstractmethod
    def grab_frame(self) -> CamFrame:
        pass

    @abstractmethod
    def get_unique_id(self) -> str:
        pass

    def get_intrinsic_matrix(self) -> np.ndarray:
        """ is called when no calibrated data is present """
        pass

    @abstractmethod
    def get_resolution(self) -> tuple[int, int]:
        """ get camera resolution in (h,w)"""
        pass

    def get_model(self):
        """" 3D model of physical camera """
        return None

    @abstractmethod
    def close(self):
        pass

    def read_camera_data(self):
        # cameratype only specifies realsense, but not which realsense -> use name from adapter
        model = self.get_model()
        if model is not None:
            self.meshes[self.name + "__model"] = model

        # load matrices from self or saved config
        self.cal_path = os.path.join(
            "data", "calibration", self.get_unique_id())
        self.img_path = os.path.join(
            "data", "images", self.get_unique_id())

        if not os.path.exists(self.cal_path):
            os.makedirs(self.cal_path)

        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        if os.path.exists(os.path.join(self.cal_path, "intrinsic_matrix.txt")):
            self.intrinsic_matrix = np.loadtxt(
                os.path.join(self.cal_path, "intrinsic_matrix.txt"))
        else:
            self.intrinsic_matrix = self.get_intrinsic_matrix()

        if os.path.exists(os.path.join(self.cal_path, "extrinsic_matrix.txt")):
            self.extrinsic_matrix = np.loadtxt(
                os.path.join(self.cal_path, "extrinsic_matrix.txt"))
        else:
            self.extrinsic_matrix = None

        if os.path.exists(os.path.join(self.cal_path, "dist_coeffs.txt")):
            self.dist_coeffs = np.loadtxt(
                os.path.join(self.cal_path, "dist_coeffs.txt"))
        else:
            self.dist_coeffs = None

        self.add_frustum()

    def add_frustum(self):
        res = self.resolution
        if res is not None:
            h, w = res
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=w, view_height_px=h, intrinsic=self.intrinsic_matrix, scale=.5, extrinsic=np.eye(4))
            self.meshes[self.name + "__frustum"] = frustum

    @property
    def resolution(self):
        return self.get_resolution()

    def point_at(self, position):
        """ point camera towards global position (blocking) """
        camera_to_target = rotation_between_vectors(
            self.pose[:3, 2], position - self.position)  # rot between camera_z and to object
        cam_pose = self.pose.copy()
        cam_pose[:3, :3] = camera_to_target.as_matrix() @ cam_pose[:3, :3]
        self.move_to_pose(cam_pose)
        self.wait_for_move(timeout=10)

    def calibrate_eyeInHand(self, ar_dict, brd):
        """ performs eye-in-hand calibration """
        intrinsic_matrix, extrinsic_matrix, dist_coeffs = self.__full_calibration(
            ar_dict, brd)
        #extrinsic_matrix = homogeneous_mat_from_RT(np.eye(3), [0, 0, 0.1])
        #intrinsic_matrix = self.intrinsic_matrix

        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.dist_coeffs = dist_coeffs

        np.savetxt(os.path.join(self.cal_path,
                   "intrinsic_matrix.txt"), intrinsic_matrix)
        np.savetxt(os.path.join(self.cal_path,
                   "extrinsic_matrix.txt"), extrinsic_matrix)
        np.savetxt(os.path.join(self.cal_path, "dist_coeffs.txt"), dist_coeffs)

        return extrinsic_matrix

    def __full_calibration(self, ar_dict, brd):

        # Units are probably wrong! [mm instead of m]

        images_directory_path = os.path.join(self.img_path)

        loaded_images = load_images(images_directory_path)
        corners, ids, sizes, markers = read_chessboards(
            loaded_images, ar_dict, brd)
        _, camera_matrix, dist_coeff, rotation_vectors, translation_vectors = cv2_calibration(
            corners, ids, sizes, brd)

        if False:
            """ show poses """
            for i, im in enumerate(loaded_images):
                frame = cv2.imread(im)
                rvec = rotation_vectors[i]
                tvec = translation_vectors[i]
                print(f"translation: {tvec}")
                annotated = cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeff, rvec, tvec, length=0.1)
                size = (np.array(frame.shape[1::-1]) / 2).astype(np.int)
                cv2.imshow("annotated", cv2.resize(annotated, size))
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        camera_poses = []
        for i in range(len(loaded_images)):
            pose_matrix = np.array(
                [translation_vectors[i], rotation_vectors[i]]).reshape(6)
            camera_poses += [pose_matrix]
            # save rotation (radians) and translation vectors (millimeters) for every pose
            # np.savetxt(path + 'data_collector/collected_data/camera_poses/pose_{}.txt'.format(i), pose_matrix, fmt='%f')

        # load robot poses from trajectory
        robot_poses = load_arrays(os.path.join(
            self.cal_path, "poses"), pattern="< >.txt")
        robot_poses = [np.loadtxt(f) for f in robot_poses]
        ret = optimize_camera_offeset(camera_poses, robot_poses)
        x = ret['x'][:6]
        extrinsic_matrix = invert_homogeneous(
            get_affine_matrix_from_6d_vector('xyz', x))

        return camera_matrix, extrinsic_matrix, dist_coeff

    def signal(self, filename=None, img: str = 'rgb'):
        """ saves img """
        if filename is None:
            filename = "{}.png".format(time_stamp())

        image_path = os.path.join(self.img_path, filename)
        frame = self.grab_frame()
        if img == 'rgb':
            cv2.imwrite(image_path, cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))

        print(f"[{time_stamp()}] Saved picture {filename}")
