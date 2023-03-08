from models.cameras.realsense import Realsense, WrongCameraException, NoCameraException
import pyrealsense2 as rs
import numpy as np


class L515(Realsense):
    rs_min_distance = 100.0
    rs_laser_power = 50.0
    rs_confidence_threshold = 2.0
    rs_receiver_gain = 10.0

    device_name = 'L515'
    rgb_shape = (1920, 1080)
    depth_shape = (1024, 768)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check for correct camera
        if self.device is None:
            raise NoCameraException

        device_product_line = str(
            self.device.get_info(rs.camera_info.product_line))
        if device_product_line != 'L500':
            raise WrongCameraException

        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'L500 Depth Sensor':
                s.set_option(rs.option.min_distance,
                             self.rs_min_distance)  # mm
                s.set_option(rs.option.laser_power, self.rs_laser_power)
                s.set_option(rs.option.confidence_threshold,
                             self.rs_confidence_threshold)
                s.set_option(rs.option.receiver_gain, self.rs_receiver_gain)

    def get_unique_id(self):
        if self.device is not None:
            return "realsense_" + str(self.device.get_info(rs.camera_info.serial_number))
        else:
            return "realsense_f0350012"
