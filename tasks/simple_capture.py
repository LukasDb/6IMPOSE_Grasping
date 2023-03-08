from .base_task import Task, TaskType
from lib.utils import time_stamp
from lib.data_acquisition import generate_training_trajectory
from lib.geometry import homogeneous_mat_from_RT

import numpy as np
import os


class CaptureTask(Task):
    name = "Data Capture"
    description = "Takes a single snapshot of RGB and robot pose for calibration."
    type = TaskType.SETUP

    def execute(self):
        app = self.app
        cam = app.camera_list[app.selected_camera]
        pose_path = os.path.join(cam.cal_path, "poses")
        if not os.path.isdir(pose_path):
            os.makedirs(pose_path)

        existing_files = os.listdir(pose_path)
        step = 0
        if len(existing_files) > 0:
            existing = [int(file.split('.')[0]) for file in existing_files]
            step = max(existing) + 1

        filename = "{:04}.png".format(step)
        cam.signal(filename)
        # save pose
        filepath = os.path.join(pose_path, "{:04}.txt".format(step))
        pose = homogeneous_mat_from_RT(
            app.robot_arm.orientation, app.robot_arm.position)
        np.savetxt(filepath, pose, fmt='%f')

        return True
