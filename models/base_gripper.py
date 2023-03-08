import numpy as np
from abc import abstractmethod
import open3d as o3d
from lib.utils import load_mesh
from lib.geometry import invert_homogeneous
from lib.gripping_pose import GrippingPose
from .targetmodel import TargetModel
from .model import Model
from models.dt_object import DTObject


class BaseGripper(TargetModel):
    """ Model of Robot arm
    the position is the EE postion and the robot is always fixed at the origin in world coordinates """
    MAX_WIDTH: float
    MAX_FORCE: float
    GRIPPER_MESH_PATH: str
    OPEN_LINK_MAT: np.ndarray
    target_width: float
    target_force: float

    def __init__(self, *args, **kwargs):
        #gripper_target_mesh = load_mesh(self.GRIPPER_MESH_PATH)
        #gripper_target_mesh.paint_uniform_color([0, 1, 0])
        #self.target = gripper_target_mesh
        super().__init__(*args, **kwargs)

        gripper_mesh = load_mesh(self.GRIPPER_MESH_PATH)
        self.meshes['gripper'] = gripper_mesh

        self.gripper_width = self.MAX_WIDTH
        self.target_width = self.MAX_WIDTH
        self.target_force = self.MAX_FORCE
        self.pre_grasp_cache = {}

    @abstractmethod
    def get_valid_grasp_depth(self, grasp_width):
        """ get valid depth of object to avoid collision with the"""
        pass

    @abstractmethod
    def get_grasp_offset_from_closed(self, grasp_width):
        """ positive valued offset of grasp depth from closed position == 0"""
        pass

    @abstractmethod
    def get_grasp_offset_mat(self, grasp_width):
        """ get offset from fully closed position(grasp_width=0) of gripper when gripper is opened at grasp_width"""
        pass

    def attach(self, obj: Model):
        link_mat = self.OPEN_LINK_MAT @ self.get_grasp_offset_mat(
            self.MAX_WIDTH)
        return super().attach(obj, link_mat)

    def get_pre_grasp_poses(self, obj: DTObject):
        if obj.name in self.pre_grasp_cache.keys():
            return self.pre_grasp_cache[obj.name]

        pre_grasps = [self.get_pre_grasp_pose(
            gpose) for gpose in obj.gripping_poses]
        self.pre_grasp_cache[obj.name] = pre_grasps
        return pre_grasps

    def get_pre_grasp_pose(self, gpose: GrippingPose):
        grasp_offset = self.get_grasp_offset_mat(gpose.width)
        return gpose.pose @ grasp_offset

    # for convenience
    def close_gripper(self):
        self.set_target_width(0)

    def open_gripper(self):
        self.set_target_width(self.MAX_WIDTH)
        # "drop"  objects
        for child in self.children:
            child.unattach()

    def set_target_width(self, width):
        self.target_width = np.clip(width, 0, self.MAX_WIDTH)

    def set_target_force(self, force):
        self.target_force = np.clip(force, 0, self.MAX_FORCE)

    def update_gripper(self, gripper_width):
        """ update gripper opening from robot controller """
        self.gripper_width = gripper_width
        if self.link_mat is not None:
            #self.link_mat[2, 3] = self.Z_CLOSED - gripper_offset
            gripper_offset = self.get_grasp_offset_mat(gripper_width)
            self.link_mat = self.OPEN_LINK_MAT @ invert_homogeneous(
                gripper_offset)
