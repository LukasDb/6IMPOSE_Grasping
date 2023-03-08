from dataclasses import dataclass
import numpy as np


@dataclass
class GrippingPose:
    pose: np.ndarray    # [4x4] matrix in local object frame
    width: float        # [mm] gripper width
