import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import math
import open3d as o3d


def distance_from_vector(matrix, seq, vector):
    return distance_from_matrices(matrix, get_affine_matrix_from_6d_vector(seq, vector))


def distance_from_matrices(p_aff, q_aff, rotation_factor=2):
    p_r = Rotation.from_matrix(p_aff[:3, :3])
    q_r = Rotation.from_matrix(q_aff[:3, :3])
    try:
        rotation_distance = np.arccos(
            np.abs(np.dot(q_r.as_quat(), p_r.as_quat())))
    except Exception:
        rotation_distance = 0
    if np.isnan(rotation_distance):
        rotation_distance = 0
    translation_distance = np.linalg.norm(
        p_aff[:3, 3] - q_aff[:3, 3])
    distance = rotation_factor * rotation_distance + translation_distance
    return distance


def invert_homogeneous(T):
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = - T[:3, :3].T @ T[:3, 3]
    return inverse


def get_affine_matrix_from_6d_vector(seq, vector):
    if seq == 'Rodriguez':
        rot, _ = cv2.Rodrigues(vector[3:])
        return homogeneous_mat_from_RT(rot, vector[:3])

    rotation = Rotation.from_euler(seq, vector[3:])
    return homogeneous_mat_from_RT(rotation, vector[:3])


def get_6d_vector_from_affine_matrix(seq, matrix):
    r_matrix = matrix[:3, :3]
    r = Rotation.from_matrix(r_matrix)
    if seq == 'Rodriguez':
        rvec = np.array(cv2.Rodrigues(r_matrix)[0]).reshape((3,))
    else:
        rvec = r.as_euler(seq)
    tvec = matrix[:3, 3]
    vec = np.concatenate((tvec, rvec))
    return np.round(vec, decimals=6)


def homogeneous_mat_from_RT(R, t):
    trans = np.eye(4)
    t = np.squeeze(t)

    if isinstance(R, Rotation):
        trans[0:3, 0:3] = R.as_matrix()
        trans[:3, 3] = t

    elif len(R) == 4 or R.shape == (4,):
        trans[0:3, 0:3] = Rotation.from_quat(R).as_matrix()
        trans[:3, 3] = t

    elif R.shape == (3, 3):
        trans[0:3, 0:3] = R
        trans[:3, 3] = t

    return trans


def rotation_between_vectors(a, b) -> Rotation:
    try:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        x = np.dot(a, b) / (a_norm * b_norm)
        angle = np.arccos(x)
    except Exception:
        return Rotation.identity()

    if abs(angle) > 1e-6:
        axis = np.cross(a/a_norm, b/b_norm)
        axis /= np.linalg.norm(axis)
        axis *= angle
        return Rotation.from_rotvec(axis)
    else:
        return Rotation.identity()
