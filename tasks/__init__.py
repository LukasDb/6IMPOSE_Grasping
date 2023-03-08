from .acquisition import AcquisitionTask
from .charboard_generation import CharBoardGenTask
from .camera_calibration import CameraCalibrationTask
from .register_gripping_pose import RegisterGrippingPoseTask
from .h2r import H2R
from .bin_picking import BinPicking
from .grasp_pose import GraspPose
from .endeffector_bbox import EndeffectorBbox
from .simple_capture import CaptureTask


task_list = [CharBoardGenTask, AcquisitionTask, CaptureTask,
             CameraCalibrationTask, RegisterGrippingPoseTask, EndeffectorBbox,
             GraspPose, BinPicking, H2R]
