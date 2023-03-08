from .base_task import Task, TaskType
from lib.grasping import get_best_gpose
from lib.trajectory import get_trajectory_step_pose
from lib.utils import time_stamp, load_mesh
from lib.geometry import homogeneous_mat_from_RT
from scipy.spatial.transform import Rotation as R
import time
import os
import math
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
import data.poses as poses


class H2R(Task):
    name = "Human-To-Robot"
    description = "Performs human to robot handover and drops object into bin"
    type = TaskType.JOB

    grip_dist = 0.1  # from which distance the robot will pick open-loop

    def execute(self):
        app = self.app
        self.obj = app.dt_objectcs_list[app.selected_dt_object]
        self.obstruction_pcd = app.obstruction_pcd

        home_pose = poses.start_h2r

        app.gripper.open_gripper()
        time.sleep(1.0)
        app.gripper.move_to_pose(home_pose)
        app.gripper.wait_for_move()
        print("Home reached.")

        gripper_path = os.path.join("data", "models", "gripper.ply")
        gripper_vis = load_mesh(gripper_path)
        gripper_vis.compute_triangle_normals()
        gripper_vis.paint_uniform_color([0.2, 0.2, 1])  # paint blue

        gripper_verts = np.load(os.path.join(
            "data", "models", "gripper_verts.npy"))
        gripper_verts[:, 2] += 0.03
        gripper_verts[:, 1] *= 0.8
        gripper_points = np.hstack(
            [gripper_verts.copy(), np.ones((len(gripper_verts), 1))])  # to homogneous

        # 0...1 for 0 is most significant
        gripper_weights = np.load(os.path.join(
            "data", "models", "gripper_weights.npy"))
        gripper_weights = 1 / np.exp(gripper_weights)

        trajectory = None
        best_gpose = None

        gripper_vis_pose = np.eye(4)

        def add_vis_to_scene():
            standard_material = rendering.MaterialRecord()
            standard_material.shader = "defaultLit"
            transformed_vis = copy.copy(gripper_vis)
            transformed_vis.transform(gripper_vis_pose)
            app._scene.scene.add_geometry(
                "grasp_pose", transformed_vis, standard_material)

            coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coor.transform(gripper_vis_pose)
            app._scene.scene.add_geometry(
                "grasp_pose_coor", coor, standard_material)

            traj_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(trajectory))
            app._scene.scene.add_geometry(
                "trajectory", traj_pcd, standard_material)

        def remove_vis_from_scene():
            if app._scene.scene.has_geometry("grasp_pose_coor"):
                app._scene.scene.remove_geometry("grasp_pose_coor")
            if app._scene.scene.has_geometry("grasp_pose"):
                app._scene.scene.remove_geometry("grasp_pose")
            if app._scene.scene.has_geometry("trajectory"):
                app._scene.scene.remove_geometry("trajectory")

        while True:
            if app.is_task_cancelled():
                gui.Application.instance.post_to_main_thread(
                    app.window, remove_vis_from_scene)
                app.robot_arm.reset()
                return

            if not self.obj.active:
                gui.Application.instance.post_to_main_thread(
                    app.window, remove_vis_from_scene)
                time.sleep(0.2)
                continue

            # get hand score
            obstruction_points = None
            if self.obstruction_pcd.pcd is not None and not self.obstruction_pcd.pcd.is_empty():
                obstruction_points = self.obstruction_pcd.get_points()

            best_gpose, best_score, ind, trajectory = get_best_gpose(
                gripper_points, gripper_weights, obstruction_points, self.grip_dist,
                self.app.gripper.get_pose(), self.obj.get_pose(), self.obj.gripping_poses)

            if best_gpose is None:
                gui.Application.instance.post_to_main_thread(
                    app.window, remove_vis_from_scene)
                continue

            dist_2_target = np.linalg.norm(
                app.gripper.pose[:3, 3] - best_gpose[:3, 3])

            print(
                f"[{time_stamp()}] Best gpose is No. {ind} with score: {best_score}; distance to target: {dist_2_target:.3f}")
            if dist_2_target < self.grip_dist:
                print(f"[{time_stamp()}] SWITCHING TO OPEN-LOOP")
                break

            next_traj = get_trajectory_step_pose(
                trajectory, app.gripper.get_pose(), best_gpose)

            app.gripper.move_to_pose(next_traj)

            gripper_vis_pose = best_gpose

            def update_vis():
                remove_vis_from_scene()
                add_vis_to_scene()
            gui.Application.instance.post_to_main_thread(
                app.window, update_vis)

        # gripping
        app.gripper.move_to_pose(best_gpose)
        print(f"[{time_stamp()}] Gripping object...")
        app.gripper.wait_for_move()
        app.gripper.close_gripper()
        print(f"[{time_stamp()}] Return to home...")
        time.sleep(3)

        # simulate gripping
        local_gpose = self.obj.gripping_poses[ind]
        offset = np.eye(4)
        offset[2, 3] = app.gripper.Z_CLOSED
        #self.obj.attach(app.robot_arm, offset @ invert_homogeneous(local_gpose))

        gui.Application.instance.post_to_main_thread(
            app.window, remove_vis_from_scene)

        pull_away = np.eye(4)
        pull_away[2, 3] = -0.05
        app.gripper.move_to_pose(best_gpose @ pull_away)

        app.gripper.wait_for_move(timeout=10)
        app.gripper.move_to_pose(home_pose)
        app.gripper.wait_for_move()
        app.gripper.open_gripper()
        time.sleep(3)
