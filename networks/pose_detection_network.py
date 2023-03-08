import tensorflow as tf
import numpy as np
import os
import cv2
import random
import open3d as o3d

from lib.utils import load_mesh
from app.app import App

from .pvn.pvn3d.data.utils import get_crop_index, pcld_processor_tf
from .pvn.pvn3d.monitor.visualizer import project2img
from .pvn.pvn3d.net.pprocessnet import InitialPoseModel


class PoseDetectionNetwork:
    step = -1

    base_crop = (80, 80)

    def __init__(self, obj_id, n_sample_points):
        self.model = None

        self.initial_pose_model = InitialPoseModel(n_point_candidate=50)

        self.n_sample_points = n_sample_points

        # load custom ops
        ops_base_path = 'networks/pvn/pvn3d/lib/pointnet2_utils/tf_ops'
        tf.load_op_library(os.path.join(
            ops_base_path, 'grouping', 'tf_grouping_so.so'))
        tf.load_op_library(os.path.join(
            ops_base_path, '3d_interpolation', 'tf_interpolate_so.so'))
        tf.load_op_library(os.path.join(
            ops_base_path, 'sampling', 'tf_sampling_so.so'))

        self.obj_id = obj_id

        model_path = os.path.join('data', 'weights', 'pvn3d', obj_id)

        self.model = tf.keras.models.load_model(model_path)

        self.model.summary()

        # ==================== load_mesh_file ====================
        kps_dir = os.path.join('data', 'models', self.obj_id)
        mesh_dir = kps_dir
        mesh_path = os.path.join(mesh_dir, "{}.ply".format(obj_id))

        self.mesh_points = np.asarray(load_mesh(mesh_path).vertices)

        kpts_path = os.path.join(kps_dir, "farthest.txt")
        corner_path = os.path.join(kps_dir, "corners.txt")
        key_points = np.loadtxt(kpts_path)
        center = [np.loadtxt(corner_path).mean(0)]
        self.mesh_kpts = np.concatenate([key_points, center], axis=0)

    def _expand_dim(self, *argv):
        item_lst = []
        for item in argv:
            item = np.expand_dims(item, axis=0)
            item_lst.append(item)
        return item_lst

    def inference(self, bbox, rgb, depth, camera_matrix, use_icp=False):
        self.step += 1

        crop_index, crop_factor = get_crop_index(
            bbox, rgb.shape[:2], base_crop_resolution=self.base_crop)

        x1, y1, x2, y2 = crop_index

        depth_crop = depth[y1:y2, x1:x2].copy()
        rgb_crop = rgb[y1:y2, x1:x2].copy()

        crop_index = (x1, y1)

        pcld_xyz, pcld_feat, sampled_index = pcld_processor_tf(
            depth_crop.astype(np.float32), rgb_crop.astype(
                np.float32)/255., camera_matrix.astype(np.float32), tf.constant(1),
            tf.constant(self.n_sample_points), tf.constant(crop_index))

        #print("pcld_process: ", time.perf_counter()- t_pcld)
        if pcld_xyz.shape[0] < self.n_sample_points:
            return False, None

        rgb_crop_resnet = tf.cast(tf.image.resize(
            rgb_crop, self.base_crop), tf.float32)

        # add batches
        rgb_crop_resnet = tf.expand_dims(rgb_crop_resnet, 0)
        pcld_xyz_ = tf.expand_dims(pcld_xyz, 0)
        pcld_feat = tf.expand_dims(pcld_feat, 0)
        sampled_index_ = tf.cast(tf.expand_dims(sampled_index, 0), tf.int32)
        crop_factor = tf.expand_dims(crop_factor, 0)

        pcld = [pcld_xyz_, pcld_feat]
        inputs = [pcld, sampled_index_, rgb_crop_resnet, crop_factor]
        kp_pre_ofst, seg_pre, cp_pre_ofst = self.model(inputs, training=False)

        R, t, kpts_voted, S = self.initial_pose_model(
            [pcld_xyz_, kp_pre_ofst, cp_pre_ofst, seg_pre, tf.expand_dims(self.mesh_kpts, 0)], training=False)

        confidence = np.linalg.norm(S.numpy(), axis=-1)[0]
        if confidence < App().settings['PoseDetection']['Score Threshold']:
            return False, None

        seg_pre = np.argmax(seg_pre[0], axis=1).squeeze()

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = tf.squeeze(t)

        if use_icp:
            #segs = seg_pre.squeeze()
            #obj_pts_index = np.where(segs == 1)[0]
            #len_index = obj_pts_index.shape[0]
            # if len_index == 0:
            #    return True, Rt, (pointcloud_obstruction_xyz, pointcloud_obstruction_rgb)
            #Rt = self.icp_refinement(Rt, pcld_xyz.numpy()[obj_pts_index])
            Rt = self.icp_refinement(Rt, pcld_xyz, pcld_feat[0][:, 3:])

        # , (pointcloud_obstruction_xyz, pointcloud_obstruction_rgb)
        return True, Rt

    def draw_obj(self, img, Rt, camera_matrix, obj_color=(255, 0, 0), camera_scale=1.0):

        # crop_index = (x1 + int((x2-x1)/2), y1+int((y2-y1)/2)) # center of bbox
        crop_index = (0, 0)

        img = project2img(self.mesh_points, Rt[:3, :], img, camera_matrix,
                          camera_scale, obj_color, crop_index) * 255
        return img.astype(np.uint8)

    def icp_refinement(self, initial_pose, pcld_xyz, normals):
        # performs significantly worse
        gpu = o3d.core.Device('CUDA:0')
        o3d_float64 = o3d.core.Dtype.Float32
        source = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(self.mesh_points, o3d_float64))
        target = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(pcld_xyz.numpy(), o3d_float64))
        target.point.normals = o3d.core.Tensor(
            normals.numpy(), o3d_float64)

        treg = o3d.t.pipelines.registration

        sigma = 0.1
        estimation = treg.TransformationEstimationPointToPlane(
            treg.robust_kernel.RobustKernel(
                treg.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))
        #estimation = treg.TransformationEstimationPointToPlane()
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.0000001,
                                               relative_rmse=0.0000001,
                                               max_iteration=30)

        max_correspondence_distance = 0.002
        voxel_size = -1
        reg_point_to_plane = treg.icp(source, target, max_correspondence_distance,
                                      initial_pose, estimation, criteria,
                                      voxel_size).transformation.numpy()
        return reg_point_to_plane
