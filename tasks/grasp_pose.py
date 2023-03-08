from .base_task import Task, TaskType
from lib.utils import load_mesh
from lib.geometry import invert_homogeneous
from models import DTObject
from scipy.spatial.transform import Rotation as R
import time
import os
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
from app.gpose_chooser import GraspPoseChooserConfig, GraspPoseChooserJob


class GraspPose(Task):
    name = "Grasp Pose"
    description = "Visualize the current best grasp pose without execution"
    type = TaskType.JOB

    grip_dist = 0.1  # from which distance the robot will pick open-loop

    def execute(self):
        app = self.app
        self.obj = app.dt_objectcs_list[app.selected_dt_object]
        self.obstruction_pcd = app.obstruction_pcd

        gripper_vis_path = os.path.join("data", "models", "gripper.ply")
        gripper_vis = DTObject(
            gripper_vis_path, name="gripper_gpose", show_axes=False)
        gripper_vis.dt_mesh.paint_uniform_color([0.2, 0.2, 1])  # paint blue

        gripper_col_path = os.path.join(
            'data', 'models', 'gripper_collision.ply')
        gripper_col = DTObject(
            gripper_col_path, name="gripper_bbox", show_axes=False, wire_frame=True)
        gripper_col.dt_mesh.paint_uniform_color([0.95, 0.05, 0.05])  # red

        config = GraspPoseChooserConfig(gripper_col_path, self.grip_dist)
        app.gpose.set_config(config)

        best_gpose = np.eye(4)

        grasp_offsets_from_closed = [app.gripper.get_grasp_offset_from_closed(
            gpose.width) for gpose in self.obj.gripping_poses]
        # convert to pre-grasp poses
        local_pre_grasps = app.gripper.get_pre_grasp_poses(self.obj)

        while not app.is_task_cancelled():
            if not self.obj.active:
                gripper_vis.remove()
                gripper_col.remove()
                time.sleep(0.2)
                continue

            obj_pose = self.obj.pose
            gripper_pose = self.app.gripper.pose

            # get hand score
            obstruction_points = self.obstruction_pcd.get_points()
            pre_grasps = [obj_pose @
                          pre_grasp for pre_grasp in local_pre_grasps]

            job = GraspPoseChooserJob(obj_pose, gripper_pose, grasp_offsets_from_closed,
                                      obstruction_points, pre_grasps)

            ind = self.app.gpose.get_gpose(job)
            if ind is None:
                gripper_vis.remove()
                gripper_col.remove()
                continue

            best_gpose = obj_pose @ self.obj.gripping_poses[ind].pose
            gripper_vis.transform(best_gpose)
            gripper_col.transform(pre_grasps[ind])
            time.sleep(0.1)

        gripper_vis.remove()
        gripper_col.remove()
        return True
