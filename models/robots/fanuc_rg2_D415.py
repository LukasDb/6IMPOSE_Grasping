from scipy.spatial.transform import Rotation as R
import os
import numpy as np
from models import OnRobot_RG2, Model
from models.targetmodel import TargetModel
from models.cameras import D415
from models.robot import Robot
from lib.utils import time_stamp, load_mesh
from app.fanuc_interface import FanucInterface


class Fanuc_RG2_D415(Robot):

    def __init__(self) -> None:
        super().__init__()
        # ----  initialize models ------
        base_path = os.path.join('data', 'models', 'fanuc_crx_10ial')

        self.gripper = OnRobot_RG2()
        self.eye_in_hand = D415(name="realsense_D415")
        robot_axes = False
        self.base = Model("fanuc_crx_base", mesh=load_mesh(
            os.path.join(base_path, 'base.stl')), show_axes=robot_axes)
        self.j1 = Model("fanuc_crx_j1", mesh=load_mesh(
            os.path.join(base_path, 'j1.stl')), show_axes=robot_axes)
        self.j2 = Model("fanuc_crx_j2", mesh=load_mesh(
            os.path.join(base_path, 'j2.stl')), show_axes=robot_axes)
        self.j3 = Model("fanuc_crx_j3", mesh=load_mesh(
            os.path.join(base_path, 'j3.stl')), show_axes=robot_axes)
        self.j4 = Model("fanuc_crx_j4", mesh=load_mesh(
            os.path.join(base_path, 'j4.stl')), show_axes=robot_axes)
        self.j5 = Model("fanuc_crx_j5", mesh=load_mesh(
            os.path.join(base_path, 'j5.stl')), show_axes=robot_axes)

        self.endeffector = TargetModel(
            "fanuc_crx", mesh=load_mesh(os.path.join(base_path, 'j6.ply')))
        self.camera_mount = TargetModel("camera_mount", mesh=load_mesh(
            os.path.join(base_path, 'd415_mount.stl')), show_axes=False)

        self.quick_changer = TargetModel("quick_changer", mesh=load_mesh(
            os.path.join(base_path, 'quick_changer.stl')), show_axes=False)

        # initialize physical robot interface
        self.robot_interface = FanucInterface(self)

        camera_mount_link_mat = np.eye(4)
        camera_mount_link_mat[2, 3] = 0.012  # from measurement of camera_mount

        # build digital twin
        self.base.transform(np.eye(4))
        self.j1.attach(self.base, np.eye(4))
        self.j2.attach(self.j1, np.eye(4))
        self.j3.attach(self.j2, np.eye(4))
        self.j4.attach(self.j3, np.eye(4))
        self.j5.attach(self.j4, np.eye(4))

        self.update_robot_visualization(np.zeros(6))

        self.camera_mount.attach(self.endeffector, np.eye(4))
        self.quick_changer.attach(self.camera_mount, camera_mount_link_mat)
        self.gripper.attach(self.endeffector)
        if self.eye_in_hand is not None:
            self.eye_in_hand.start()
            self.eye_in_hand.attach(
                self.endeffector, self.eye_in_hand.extrinsic_matrix)

    def update_robot_visualization(self, joint_pose):
        # Fanuc mixes J2 and J3
        joint_pose[2] = joint_pose[2] + joint_pose[1]

        j1_pose = np.eye(4)
        j2_pose = np.eye(4)
        j3_pose = np.eye(4)
        j4_pose = np.eye(4)
        j5_pose = np.eye(4)
        j1_pose[:3, :3] = R.from_euler(
            'z', [joint_pose[0]], degrees=True).as_matrix()
        j1_pose[2, 3] = 0.085
        j2_pose[:3, :3] = R.from_euler(
            'XZ', [90, -joint_pose[1]], degrees=True).as_matrix()
        j2_pose[2, 3] = 0.16
        j3_pose[:3, :3] = R.from_euler(
            'Z', [joint_pose[2]], degrees=True).as_matrix()
        j3_pose[1, 3] = 0.71

        j4_pose[:3, :3] = R.from_euler(
            'YZ', [90, -joint_pose[3]], degrees=True).as_matrix()
        j4_pose[0, 3] = 0.54

        j5_pose[:3, :3] = R.from_euler(
            'YZ', [-90, joint_pose[4]], degrees=True).as_matrix()
        j5_pose[0, 3] = -0.15

        self.j1.link_mat = j1_pose
        self.j2.link_mat = j2_pose
        self.j3.link_mat = j3_pose
        self.j4.link_mat = j4_pose
        self.j5.link_mat = j5_pose

        base_pose = np.eye(4)
        base_pose[2, 3] = -0.245
        self.base.transform(base_pose)

    def start(self) -> None:
        self.robot_interface.start()

    def close(self) -> None:
        self.robot_interface.close()

    def set_simulate(self, state) -> None:
        self.robot_interface.set_simulate(state)

    def set_lock(self, state) -> None:
        self.robot_interface.set_lock(state)

    def reset(self) -> None:
        self.robot_interface.reset()

    @property
    def online(self):
        return self.robot_interface.online
