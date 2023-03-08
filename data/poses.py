import numpy as np
from lib.geometry import homogeneous_mat_from_RT
from scipy.spatial.transform import Rotation as R

# BIN PICKING
_orientation = np.array(
    [-179/180*np.pi, -0/180*np.pi, 84/180*np.pi])  # bin picking
_position = np.array([0.6, 0.0, 0.4])  # bin picking

start_bin_pick = homogeneous_mat_from_RT(
    R.from_euler('xyz', _orientation), _position)

# HUMAN 2 ROBOT
_orientation = np.array(
    [165/180*np.pi, -85/180*np.pi, 70/180*np.pi])
_position = np.array([0.26, 0.53, 0.05])

start_h2r = homogeneous_mat_from_RT(
    R.from_euler('xyz', _orientation), _position)
