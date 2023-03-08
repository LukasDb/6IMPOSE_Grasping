from .base_gripper import BaseGripper
import math
import os
import numpy as np


class OnRobot_RG2(BaseGripper):
    def __init__(self, *args, **kwargs):
        self.MAX_WIDTH = 0.101  # as reported by the plugin
        self.MAX_FORCE = 30
        self.GRIPPER_MESH_PATH = os.path.join("data", "models", "gripper.ply")

        self.Z_OPEN = 0.2                 # z offset when opened
        self.OPEN_LINK_MAT = np.eye(4)
        self.OPEN_LINK_MAT[2, 3] = self.Z_OPEN
        super().__init__(*args, **kwargs, name="OR_RG2")

    def get_grasp_offset_from_closed(self, grasp_width):
        widths = np.array([0.0, 0.01,  0.02, 0.03, 0.04, 0.05,
                           0.06, 0.07, 0.08, 0.09, 0.1, 0.101])
        depths = np.array([0.0, 0.5,   1.1,   3.0,  5.0,  6.6,
                           10.9, 13.7, 20.3, 27.6, 36.8, 39.5]) / 1000.
        grasp_offset = np.interp(grasp_width, widths, depths)

        return grasp_offset

    def get_valid_grasp_depth(self, grasp_width):
        """ get valid depth of object to avoid collision with the grasper"""
        # from datasheet
        widths = np.arange(0.01, 0.11, 0.01)
        depths = np.array([0.065, 0.05, 0.05, 0.049, 0.048,
                          0.045, 0.042, 0.04, 0.039, 0.036])
        return np.interp(grasp_width, widths, depths)

    def get_grasp_offset_mat(self, grasp_width):
        grasp_width = np.clip(grasp_width, 0, self.MAX_WIDTH)
        grasp_offset_mat = np.eye(4)
        offset = self.get_grasp_offset_from_closed(grasp_width)
        # because from opened gripper
        grasp_offset_mat[2, 3] = offset - 0.0395
        return grasp_offset_mat
