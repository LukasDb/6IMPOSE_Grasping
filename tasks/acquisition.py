from .base_task import Task, TaskType
from lib.utils import time_stamp
from lib.data_acquisition import generate_training_trajectory
from lib.geometry import homogeneous_mat_from_RT

import numpy as np
import os


class AcquisitionTask(Task):
    name = "Data Acquisition"
    description = "Starts the data acquisition with the settings defined in config/settings.py."
    type = TaskType.SETUP

    def execute(self):
        app = self.app
        trajectory = generate_training_trajectory()
        cam = app.camera_list[app.selected_camera]
        pose_path = os.path.join(cam.cal_path, "poses")
        if not os.path.isdir(pose_path):
            os.makedirs(pose_path)

        print(f"[{time_stamp()}] Acquiring data from camera: {app.selected_camera}")

        for step, traj in enumerate(trajectory):
            app.robot_arm.move_to_pose(traj)
            if not app.robot_arm.wait_for_move(cb=app.is_task_cancelled):
                break
            # save img
            filename = "{:04}.png".format(step)
            cam.signal(filename)
            # save pose
            filepath = os.path.join(pose_path, "{}.txt".format(step))
            pose = homogeneous_mat_from_RT(
                app.robot_arm.orientation, app.robot_arm.position)
            np.savetxt(filepath, pose, fmt='%f')

        return True
