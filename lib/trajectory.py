import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from .geometry import homogeneous_mat_from_RT, rotation_between_vectors
from scipy.special import comb


def get_trajectory_step_pose(trajectory, current_pose, target_pose):
    current_pos = current_pose[:3, 3]
    dist_2_target = np.linalg.norm(current_pos - target_pose[:3, 3])
    move_target = get_trajectory_step_pos(
        current_pos, trajectory, dist_2_target)
    orientation_target = get_trajectory_step_orn(
        current_pose, move_target, target_pose)
    next_traj = homogeneous_mat_from_RT(orientation_target, move_target)
    return next_traj


def get_trajectory_step_orn(current_pose, trajectory_position, target_pose):
    point_towards_target = target_pose[:3, 3] - trajectory_position
    point_towards_target = rotation_between_vectors(
        current_pose[:3, 2], point_towards_target)

    point_towards_target = point_towards_target.as_matrix()

    # apply to current orientation
    point_towards_target = point_towards_target @ current_pose[:3, :3]

    rel_target_orientation = point_towards_target[:3,
                                                  :3].T @ target_pose[:3, :3]
    rel_target_x_vec = rel_target_orientation[:3, 0]
    # smoothly approach
    rel_rotation = math.atan2(rel_target_x_vec[1], rel_target_x_vec[0])
    rel_z_rotation = R.from_euler('xyz', [0., 0., rel_rotation]).as_matrix()

    move_orientation = point_towards_target @ rel_z_rotation

    return move_orientation


def get_trajectory_step_pos(gripper_position, trajectory, dist_2_target):
    """ find the move command pose according to
            - minimum step length
            - maximum step length
            - the closer the smaller steps
    """
    # tweak this
    wanted_move_dist = dist_2_target / 2.0
    wanted_move_dist = np.clip(wanted_move_dist, 0.02, 0.1)

    # choose trajectory point to fulfill wanted move dist
    # last trajectory point is the current robot pos
    previous = trajectory[-1]
    for point in trajectory[::-1]:
        if np.linalg.norm(gripper_position - point) > wanted_move_dist:
            return previous

        previous = point
    return None


def generate_traj(start_pose, target_pose, tangent_weight=0.2):
    # add control point at negative z from target_pose
    control_point = np.array([[0, 0, 0, 1.0]]).T
    control_point[2] = - tangent_weight
    control_point = target_pose @ control_point
    control_point = control_point[:3, 0].T.squeeze()

    target = target_pose[:3, 3]
    cpoints = np.stack([start_pose[:3, 3], control_point, target], axis=0)
    return bezier_curve(cpoints)


def bernstein_poly(i, n, t):
    """
        The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=100):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1], 
                [2,3], 
                [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = points[:, 0]
    yPoints = points[:, 1]
    zPoints = points[:, 2]

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    zvals = np.dot(zPoints, polynomial_array)

    return np.stack([xvals, yvals, zvals], axis=0).T
