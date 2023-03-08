from .base_task import Task, TaskType
from lib.utils import time_stamp
from lib.geometry import invert_homogeneous, homogeneous_mat_from_RT


class RegisterGrippingPoseTask(Task):
    name = "Gripping Pose Registration"
    description = "Saves the current pose relative to the selected object as a possible grasping pose for this object."
    type = TaskType.SETUP

    def execute(self):
        app = self.app
        dtobj = app.dt_objectcs_list[app.selected_dt_object]

        if not dtobj.active:
            print(f"[{time_stamp()}] Pose of {dtobj.name} unknown!")
            return

        cur_robot = app.gripper.pose
        cur_object = dtobj.pose
        robot_in_object_frame = invert_homogeneous(cur_object) @ cur_robot
        dtobj.register_gripping_pose(robot_in_object_frame)

        return True
