from abc import abstractmethod
from models.cameras.base_camera import Camera, CamFrame
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from app.app import App

from lib.utils import load_mesh, time_stamp


class WrongCameraException(Exception):
    pass


class NoCameraException(Exception):
    pass


class Realsense(Camera):
    @property
    @abstractmethod
    def rgb_shape(self):
        pass

    @property
    @abstractmethod
    def depth_shape(self):
        pass

    @property
    @abstractmethod
    def device_name(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.pipeline = None
        try:
            self.pipeline = rs.pipeline()
            self.config = config = rs.config()

            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            self.device = pipeline_profile.get_device()
        except Exception:
            self.pipeline = None
            self.read_camera_data()
            return

    def start(self):
        if self.device is None:
            return

        self.config.enable_stream(rs.stream.depth, *self.depth_shape,
                                  rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, *self.rgb_shape, rs.format.bgr8, 30)
        self.align_to_rgb = rs.align(rs.stream.color)

        s = App().settings['Realsense']
        s_spat = s['spatial']
        s_temp = s['temporal']

        self.temporal_filter = rs.temporal_filter(
            smooth_alpha=s_temp['alpha'], smooth_delta=s_temp['delta'], persistence_control=2)

        self.spatial_filter = rs.spatial_filter(
            smooth_alpha=s_spat['alpha'], smooth_delta=s_spat['delta'], magnitude=s_spat['magnitude'], hole_fill=s_spat['hole_fill'])

        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

      # Start streaming
        self.pipeline.start(self.config)
        self.read_camera_data()

    def get_resolution(self) -> tuple[int, int]:
        return self.rgb_shape[::-1]

    def get_model(self):
        return load_mesh(os.path.join("data", "models", f"realsense_{self.device_name}.ply"))

    def grab_frame(self) -> CamFrame:
        if self.device is None:
            return None
        output = CamFrame()
        frames = self.pipeline.wait_for_frames()
        # makes depth frame same resolution as rgb frame
        frames = self.align_to_rgb.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            print("Could not get camera frame")
            return output

        s = App().settings['Realsense']
        if s["Update"]:
            try:
                self.temporal_filter.set_option(
                    rs.option.filter_smooth_alpha, s_temp['alpha'])
                self.temporal_filter.set_option(
                    rs.option.filter_smooth_delta, s_temp['delta'])

                self.spatial_filter.set_option(
                    rs.option.filter_smooth_alpha, s_spat['alpha'])
                self.spatial_filter.set_option(
                    rs.option.filter_smooth_delta, s_spat['delta'])
                self.spatial_filter.set_option(
                    rs.option.filter_magnitude, s_spat['magnitude'])
                self.spatial_filter.set_option(
                    rs.option.holes_fill, s_spat['hole_fill'])
            except Exception as e:
                print(f"[{time_stamp()}] {e}")

        s_spat = s['spatial']
        s_temp = s['temporal']

        if s['temporal_filter']:
            depth_frame = self.temporal_filter.process(depth_frame)

        if s['spatial_filter']:
            depth_frame = self.spatial_filter.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data()).astype(
            np.float32) * self.depth_scale
        color_image = cv2.cvtColor(np.asanyarray(
            color_frame.get_data()), cv2.COLOR_BGR2RGB)

        output.depth = depth_image
        output.rgb = color_image
        output.extrinsic = self.pose
        output.intrinsic = self.intrinsic_matrix

        return output

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
