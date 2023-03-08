import os
import open3d as o3d
import numpy as np
from datetime import datetime
import json
import sys
from lib.gripping_pose import GrippingPose


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def time_stamp():
    return datetime.now().strftime("%H:%M:%S:%f")


def get_gripper_vis(gripper, gpose: GrippingPose, thickness=0.005):
    grasp_width = gpose.width
    gripper_height = gripper.get_valid_grasp_depth(grasp_width)
    f_h = thickness      # finger height
    f_d = thickness      # fingler thickness
    f_over = 0.  # finger length over grasp
    y_pos_finger = o3d.geometry.TriangleMesh.create_box(
        f_h, f_d, gripper_height + f_over)
    y_neg_finger = o3d.geometry.TriangleMesh.create_box(
        f_h, f_d, gripper_height + f_over)
    palm = o3d.geometry.TriangleMesh.create_box(f_h, grasp_width + 2*f_d, f_d)
    stem = o3d.geometry.TriangleMesh.create_box(
        f_h, f_h, 0.05)

    y_pos_finger.translate([-f_h/2, grasp_width/2, -gripper_height])
    y_neg_finger.translate([-f_h/2, -grasp_width/2 - f_d, -gripper_height])
    stem.translate([-f_h/2, -f_h/2, -0.05-gripper_height-f_d])
    palm.translate([-f_h/2, -grasp_width/2.-f_d, -gripper_height-f_d])

    gripper_vis = y_neg_finger + y_pos_finger + palm + stem
    gripper_vis.compute_vertex_normals()
    gripper_vis.transform(gpose.pose)
    return gripper_vis


def load_mesh(meshpath, tensor=False, debug=False):
    if not tensor:
        mesh = o3d.io.read_triangle_mesh(meshpath)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(
            allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        self_intersecting = mesh.is_self_intersecting()
        watertight = mesh.is_watertight()
        orientable = mesh.is_orientable()

        if debug:
            #print(f"[{time_stamp()}] Reading mesh: {meshpath}")
            print(f"  edge_manifold:          {edge_manifold}")
            print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
            print(f"  vertex_manifold:        {vertex_manifold}")
            print(f"  self_intersecting:      {self_intersecting}")
            print(f"  watertight:             {watertight}")
            print(f"  orientable:             {orientable}")
    else:
        mesh = o3d.t.io.read_triangle_mesh(meshpath)

    try:
        with open(os.path.join(os.path.dirname(meshpath), "metadata.json")) as F:
            meta = json.load(F)

        if 'scale' in meta.keys():
            mesh.scale(scale=meta['scale'], center=[0., 0., 0.])
        if 'color' in meta.keys():
            mesh.paint_uniform_color(meta['color'])

    except Exception:
        print(f"[{time_stamp()}] Did not find metadata for {meshpath}")
        pass

    return mesh
