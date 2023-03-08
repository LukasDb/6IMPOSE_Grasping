import numpy as np
from lib.geometry import get_affine_matrix_from_6d_vector
from app.app import App


def generate_training_trajectory():
    """ from fieldview-client """
    # TODO compensate for matrix - tool difference by applying the extrinsic matrix calibration
    # TODO compensate for the camera position and physical limits of the tool by rotating the z axis in order to reach angles that are otherwise not reachable.
    # TODO reject positions that are not reachable for the robot.
    sett = App().settings['Acquisition Trajectory']
    return spherical_trajectory(sett['center'], sett['radius'], sett['jitter'], sett['steps'], sett['max_angle'])


def spherical_trajectory(center, radius, jitter, angle_steps, max_angle):
    # world coordinates
    center_xyzatp = np.zeros(6, np.float32)
    center_xyzatp[:3] = center
    center_array = get_affine_matrix_from_6d_vector('xyz', center_xyzatp)
    # move on #angle_steps circles between over the object and at max_angle
    angles = np.arange(max_angle/angle_steps, max_angle, max_angle/angle_steps)
    # find a point on the circle of fixed height with fixed distance from the center
    # we start from the center in camera coordinates
    rotation = np.zeros((angles.shape[0], 6), np.float32)
    rotation[:, 3] = angles
    base_positions = np.array([0, 0, radius, np.pi, 0,  0])[
        np.newaxis, :] + rotation
    circle_steps = np.ceil(angle_steps * np.sin(angles) / np.sin(max_angle))
    trajectory = []
    for i in range(base_positions.shape[0]):
        for a in np.arange(np.pi*2/circle_steps[i], np.pi*2, np.pi*2/circle_steps[i]):
            new_pos = base_positions[i, :]
            new_pos[4] = a
            rand = np.random.rand(6) * 2.0 - 1.0
            rand = rand * jitter
            new_pos = new_pos + rand
            new_pos[5] = 0.0
            new_pos_matrix = get_affine_matrix_from_6d_vector('XZX', new_pos)
            final_matrix = center_array @ np.linalg.inv(new_pos_matrix)
            x_axis = np.array([1, 0, 0, 1])
            p_x_axis = final_matrix @ x_axis
            origin = np.array([0, 0, 0, 1])
            p_origin = final_matrix @ origin
            x_v = p_x_axis - p_origin
            angle = np.arccos(
                x_v[1]/(x_v[0] ** 2 + x_v[1] ** 2)) * np.sign(x_v[0]) - np.pi/2
            correction = get_affine_matrix_from_6d_vector(
                'zyx', np.array([0, 0, 0, angle, 0, 0]))
            final_matrix = final_matrix @ np.linalg.inv(correction)

            #fanuc_vector = get_affine_matrix_from_6d_vector('xyz', np.array([0,0,0,0,0,]), degrees=True)
            #fanuc_vector[5] = (np.random.rand(1) * 2.0 - 1.0) * jitter[5]
            #final_matrix = get_affine_matrix_from_6d_vector('xyz', fanuc_vector, degrees=True)

            trajectory.append(final_matrix)
    return trajectory
