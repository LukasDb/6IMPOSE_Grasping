import json
from math import degrees
import math
import os
import numpy as np
import open3d as o3d
from lib.geometry import distance_from_matrices, homogeneous_mat_from_RT
from lib.utils import time_stamp, load_mesh
from lib.gripping_pose import GrippingPose
from models.targetmodel import TargetModel
import copy
from scipy.spatial.transform import Rotation as R
from typing import List


class DTObject(TargetModel):
    """ Digital Twin Object"""
    gripping_poses: List[GrippingPose]

    def __init__(self, meshpath, wire_frame=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        geometry = load_mesh(meshpath)

        if wire_frame:
            geometry = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)

        self.data_path = os.path.dirname(meshpath)
        self.grip_path = os.path.join(self.data_path, "gripping_poses.json")

        self.meshes[self.name + '__dt_object'] = geometry

        self.gripping_poses = []
        self._bounding_box = None

        try:
            self._read_gpose()
        except Exception as e:
            print(
                f"[{time_stamp()}] Could not read grasp poses for {self.name}: {e}")

        self.hide()

    @property
    def bounding_box(self):
        """ bounding box corners in local object frame: [8,3]"""
        if self._bounding_box is None:
            self._bounding_box = self._get_bounding_box()
        return self._bounding_box

    @property
    def dt_mesh(self):
        return self.meshes[self.name + '__dt_object']

    def delete_gripping_poses(self):
        self.gripping_poses.clear()
        self._write_gpose()

    def register_gripping_pose(self, pose, correct_width=True, grasp_width=None, write=True):
        """ registers pose as gripping pose in dt object coordinate frame """

        if correct_width or grasp_width is None:
            obj_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.dt_mesh)
            # Create a scene and add the triangle mesh
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(obj_mesh_t)
            start_0 = pose[:3, 3]
            start_1 = start_0.copy()
            pos_y = pose[:3, 1]
            pos_y /= np.linalg.norm(pos_y)
            neg_y = - pos_y

            # THIS DOES NOT WORK WITH CONCAVE MESHES
            rays = o3d.core.Tensor([[*start_0, *pos_y], [*start_1, *neg_y]],
                                   dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays)
            dists = ans['t_hit'].numpy()

            if np.inf in dists:
                print(f"[{time_stamp()}] Grasp Pose outside of mesh!")
                return
            if len(dists) != 2:
                print(ans)
                print(dists)
                raise AssertionError

            y_correction = (dists[0] - dists[1]) / 2.0
            y_corr_mat = np.eye(4)
            y_corr_mat[1, 3] = y_correction
            grasp_width = float(dists[0] + dists[1])

            pose = pose @ y_corr_mat

        gpose = GrippingPose(pose, grasp_width)

        self.gripping_poses.append(gpose)

        if write:
            self._write_gpose()

    def __overwrite_gripping_pose(self, pose, pose_ind):
        # legacy
        self.gripping_poses[pose_ind] = pose
        self._write_gpose()

    def _write_gpose(self):
        json_store = [{"pose": np.array2string(
            g.pose, separator=','), "width": g.width} for g in self.gripping_poses]
        with open(self.grip_path, 'w') as F:
            json.dump(json_store, F, indent="\t")

    def _read_gpose(self):
        if os.path.isfile(self.grip_path):
            with open(self.grip_path) as F:
                json_load = json.load(F)
            self.gripping_poses = [
                GrippingPose(eval('np.array(' + x["pose"] + ')'), x["width"]) for x in json_load]

    def _get_bounding_box(self):
        bbox = self.dt_mesh.get_axis_aligned_bounding_box()
        return np.asarray(bbox.get_box_points())
