import open3d.visualization.gui as gui
from app.app import App
import time
import numpy as np
import open3d as o3d
from models.model import Model
from models import DTObject
from models.cameras.base_camera import CamFrame


class ObstructionPcd(Model):
    """ Hand Pointcloud """

    def __init__(self, *args, **kwargs):
        super().__init__(name="ObstructionPCD", show_axes=False,
                         *args, **kwargs)
        self.pcd = o3d.t.geometry.PointCloud()
        self.meshes[self.name + '__pcd'] = self.pcd
        #self.pcds = {}

    def _update_transform(self):
        # pointcloud update does not trigger GUI updates
        self._remove()
        return super()._update_transform()

    def update_from_scene(self, frame: CamFrame, model: DTObject, obj_pose_in_cam):
        """ update obstruction pointcloud from a CamFrame frame and removes the points belonging to an object """
        settings = App().settings['Obstruction PointCloud']
        update_pcd = self.get_pcd_from_rgbd(frame, settings)
        update_pcd.transform(frame.extrinsic)

        # This snippet adds a fake pcl for debugging
        """
        x1, x2 = -0.1, 0.1
        y1, y2 = -0.05, 0.1
        z1, z2 = 0.4, 0.405
        n = 40
        x, y, z = np.meshgrid(np.linspace(x1, x2, n), np.linspace(
            y1, y2, n), np.linspace(z1, z2, 2))
        points = o3d.core.Tensor(
            np.stack((x, y, z), axis=-1).reshape((-1, 3)), dtype=o3d.core.Dtype.Float32)
        pcd.point.positions = o3d.core.concatenate(
            (pcd.point.positions, points), axis=0)
        """

        # test "persistent" mode not so efficient
        #t = time.perf_counter()
        # self.pcds = {k: v for k, v in self.pcds.items()
        #             if time.perf_counter()-k < settings['persistency']}

        #pcd = update_pcd.clone()
        # for old_pcd in self.pcds.values():
        #    pcd = pcd.append(old_pcd)
        #self.pcds.update({time.perf_counter(): update_pcd})
        #pcd = pcd.voxel_down_sample(0.005)
        #print(f"persistent: {time.perf_counter() - t}")

        pcd = update_pcd

        if obj_pose_in_cam is not None:
            points, colors = self.remove_points_inside_model(
                pcd, model, frame.extrinsic @ obj_pose_in_cam, settings)
        else:
            points = pcd.point.positions
            colors = pcd.point.colors

        self.pcd.point.positions = points
        self.pcd.point.colors = colors

        if self.window is not None:
            gui.Application.instance.post_to_main_thread(
                self.window, self._update_transform)

    def get_pcd_from_rgbd(self, frame: CamFrame, settings):
        gpu = o3d.core.Device('CUDA:0')

        rgb = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(
            frame.rgb))#.to(gpu)
        depth = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(
            frame.depth))#.to(gpu)
        rgbd = o3d.t.geometry.RGBDImage(rgb, depth)
        intr_mat = o3d.core.Tensor(frame.intrinsic)

        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intr_mat, depth_scale=1.0, stride=max(settings['stride'], 1))

        vox_size = App().settings['Obstruction PointCloud']['voxel size']
        if vox_size > 0.0:
            pcd = pcd.voxel_down_sample(vox_size)
        return pcd.cpu()

    def remove_points_inside_model(self, pcd_t, model: DTObject, obj_pose_in_cam, settings):

        # IDEA dont calculate for all points only close ones (oct tree? pre crop?)

        scene = o3d.t.geometry.RaycastingScene()
        mesh = model.dt_mesh
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_t.transform(obj_pose_in_cam)
        _ = scene.add_triangles(mesh_t)

        result = scene.compute_signed_distance(
            pcd_t.cpu().point.positions).numpy()  # only on cpu
        result = result > settings['remove_tolerance']

        # a little bit faster but no tolerance and scaling doesnt really work
        #result = scene.compute_occupancy(pcd_t.cpu().point.positions).numpy()
        #result = result < 0.5
        points = pcd_t.point.positions[result]
        colors = pcd_t.point.colors[result]
        return points, colors

    def get_points(self):
        """ points in global coordinate frames """
        if self.pcd.is_empty():
            return None
        return self.pcd

    def clear(self):
        if self.pcd is not None:
            self.pcd.clear()
