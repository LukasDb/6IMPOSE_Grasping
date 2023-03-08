from .base_task import Task, TaskType
import open3d as o3d
import open3d.visualization.gui as gui
from lib.utils import time_stamp
from lib.geometry import invert_homogeneous
import numpy as np
import copy


class EndeffectorBbox(Task):
    name = "Endeffector Bbox"
    description = "Calculate the bounding box (collision mesh) for the endeffector assembly"
    type = TaskType.SETUP

    def execute(self):
        app = self.app

        def add_mesh(endeffector, obj, link_mat=np.eye(4)):
            for name, mesh_ in obj.meshes.items():
                if '__axes' in name:
                    continue

                if '__frustum' in name:
                    tris = np.array([
                        [0, 1, 2],
                        [0, 2, 3],
                        [0, 3, 4],
                        [0, 4, 1],
                        [1, 2, 3],
                        [1, 3, 4]
                    ])

                    mesh = o3d.geometry.TriangleMesh()
                    mesh. vertices = mesh_.points
                    mesh.triangles = o3d.utility.Vector3iVector(tris)
                else:
                    mesh = copy.copy(mesh_)

                print(f"[{time_stamp()}] Adding mesh", name)
                mesh.transform(link_mat @ obj.link_mat)
                endeffector += mesh

            # recursively add children of children
            for child in obj.children:
                add_mesh(endeffector, child, obj.link_mat)

        # dont add robot arm itself
        endeffector = o3d.geometry.TriangleMesh()
        for child in app.robot_arm.children:
            add_mesh(endeffector, child)

        # endeffector.transform(invert_homogeneous(app.gripper.pose))

        filename = "out.ply"
        o3d.io.write_triangle_mesh(filename, endeffector)
        return True
