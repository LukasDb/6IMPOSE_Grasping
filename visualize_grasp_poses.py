import copy
import open3d as o3d
import os
from models.dt_object import DTObject
from models.grippers import OnRobot_RG2
from lib.utils import get_gripper_vis
import begin


@begin.start
def main(obj_id, all=False):
    global current, gripper_mesh, gripper_frame

    obj = DTObject(meshpath=os.path.join("data", "models", obj_id,
                                         f"{obj_id}.ply"), show_axes=True, name='object')

    gripper = OnRobot_RG2()

    obj_mesh = obj.dt_mesh
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    print("\nTotal grasp poses: ", len(obj.gripping_poses))

    if all:
        grippers = []
        for grasp_pose in obj.gripping_poses:
            gripper_vis = get_gripper_vis(gripper, grasp_pose, thickness=0.001)
            grippers.append(gripper_vis)
        o3d.visualization.draw_geometries([origin, obj_mesh, *grippers])

    else:
        print("Press [k] to show the next grasp pose")
        gripper_vis = get_gripper_vis(gripper, obj.gripping_poses[0])

        def test(vis):
            test.current += 1
            test.current = test.current % len(obj.gripping_poses)

            vis.remove_geometry(test.gripper_vis, reset_bounding_box=False)
            vis.remove_geometry(test.gripper_frame, reset_bounding_box=False)
            
            gpose = obj.gripping_poses[test.current]
            test.gripper_vis = get_gripper_vis(gripper, gpose)
            test.gripper_frame = copy.copy(origin)
            test.gripper_frame.transform(gpose.pose)
            print(
                f"Visualizing Grasp pose {test.current} with grasp width: {gpose.width:.4f}")

            vis.add_geometry(test.gripper_vis, reset_bounding_box=False)
            vis.add_geometry(test.gripper_frame, reset_bounding_box=False)

        test.current = 0
        test.gripper_frame = copy.copy(origin)
        test.gripper_vis = gripper_vis

        key_to_callback = {}
        key_to_callback[ord("K")] = test

        o3d.visualization.draw_geometries_with_key_callbacks(
            [origin, obj.dt_mesh, gripper_vis], key_to_callback)
