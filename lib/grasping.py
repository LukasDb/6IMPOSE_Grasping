from scipy.spatial.distance import cdist
import numpy as np
from lib.geometry import distance_from_matrices, invert_homogeneous
from lib.utils import time_stamp
from .trajectory import generate_traj
import open3d as o3d
import math
from typing import List
from dataclasses import dataclass
import copy


@dataclass
class GraspPoseChooserConfig:
    gripper_collision_path: str
    grip_distance: float


@dataclass
class GraspPoseChooserJob:
    obj_pose: np.ndarray  # in global frame
    gripper_pose: np.ndarray  # in global frame
    grasp_offsets: List[float]  # in m from closed position
    obstruction_pcl: o3d.t.geometry.PointCloud  # in global frame
    gripping_poses: List[np.ndarray]  # in global frame


def choose_best_grasp_pose(job: GraspPoseChooserJob, scene, grip_dist):
    if job.obstruction_pcl is None:
        return None

    inds = range(len(job.gripping_poses))

    # filter gposes "that point away"
    # this avoids grasping the object from "behind"
    poses = job.gripping_poses
    gripper_z = job.gripper_pose[:3, 2]

    rel_angle = [np.dot(gripper_z, gpose[:3, 2]) for gpose in poses]
    angle_thr = np.cos(60/180*np.pi)
    inds = [x for x in inds if rel_angle[x] > angle_thr]

    # find distances to obstruction pointcloud

    best_i = None
    best_score = math.inf

    for i in inds:
        grasp_depth = job.grasp_offsets[i]
        gpose = poses[i]

        pcd = job.obstruction_pcl.clone()
        pcd.transform(invert_homogeneous(gpose))
        pcd = pcd.select_by_mask(o3d.core.Tensor(
            pcd.point.positions.numpy()[:, 2] < - grasp_depth+0.02))  # dont consider points that "are further away" + finger_over tolerance
        # transform world pcl into local gpose frame
        #import pickle
        #p = pcd.point.positions.numpy()
        #pickle.dump(p, open("points", 'wb'))
        #pickle.dump(gpose, open("gpose", 'wb'))
        #pickle.dump(job.obj_pose, open("obj_pose", 'wb'))
        #print("updated files")
        # exit()
        result = scene.compute_signed_distance(pcd.point.positions).numpy()
        n_points = np.count_nonzero(result < 0.001)
        if n_points < 1:
            dist_score = np.mean(1. / result)
            if dist_score < best_score:
                best_i = i
                best_score = dist_score

    return best_i

    # choose the shortest trajectory
    dists = []
    for i in inds:
        traj = generate_traj(job.gripper_pose, job.obj_pose @
                             job.gripping_poses[i], tangent_weight=grip_dist-0.01)
        starts = traj[:-1, :]
        ends = traj[1:, :]
        dists.append(np.sum([np.linalg.norm(x-y)
                     for x, y in zip(starts, ends)]))
    #i_shortest = np.argmin(dists)
    dists = np.array(dists)
    short_score = 1 / dists
    short_score /= np.max(short_score)

    overall_score = safe_score + short_score
    best = np.argmax(overall_score)

    return inds[i_safest]


def get_best_gpose(gripper_points, gripper_weights, obstruction_points, grip_dist, gripper_pose, obj_pose, gripping_poses):

    # get hand score
    results = [
        eval_grip_pose(gripper_pose, obj_pose @ gpose,
                       grip_dist, obstruction_points, gripper_points, gripper_weights)
        for gpose in gripping_poses
    ]  # [ (scores, traj)]

    scores, trajs = list(zip(*results))

    # weights = np.array([0.2,0.008])

    weights = np.array([1, 50.])

    # normalize scores
    scores = np.array(scores)  # [n, 2] list of scores
    weighted_scores = scores * weights
    weighted_scores_acc = np.sum(weighted_scores, axis=1)

    ind = np.argmax(weighted_scores_acc)

    best_score = weighted_scores[ind]
    best_traj = trajs[ind]
    best_gpose = obj_pose @ gripping_poses[ind]

    if np.sum(best_score) > 4.4:
        return best_gpose, best_score, ind, best_traj
    else:
        print(f"[{time_stamp()}] Could not find pose with good enough score ({np.sum(best_score)}: {best_score})")
        return None, best_score, None, None


def eval_grip_pose(start_pose, grip_pose: np.ndarray, grip_dist, obstruction_points, gripper_points, gripper_point_weights):
    """ideas:
        - take the minium distance from the actual gripper mesh to the pcd
            -> the larger the better
    """
    # prefer solutions with shorter trajectory (higher score)
    # TODO number of trajectory points should depend on target_distance
    traj = generate_traj(start_pose, grip_pose, tangent_weight=grip_dist-0.01)
    starts = traj[:-1, :]
    ends = traj[1:, :]

    scores = [-1, -1]  # move, hand
    # negative initialization makes "incomplete" scores worse than complete ones

    traj_length = np.sum([np.linalg.norm(x-y) for x, y in zip(starts, ends)])
    # scores[0] = 1.0 / traj_length
    scores[0] = 1. / (traj_length + 0.01 *
                      distance_from_matrices(grip_pose, start_pose, rotation_factor=4))

    gripper_verts = (grip_pose @ gripper_points.T)[:3].T

    if obstruction_points is not None and len(obstruction_points) > 0:
        dists = cdist(gripper_verts, obstruction_points)
        dists = np.min(dists, axis=1) * gripper_point_weights
        scores[1] = np.mean(dists)
    else:
        scores[1] = 0.

    return scores, traj
