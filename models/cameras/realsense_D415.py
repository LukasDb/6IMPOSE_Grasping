import pyrealsense2 as rs
import numpy as np
from .realsense import Realsense
import os
import json


class D415(Realsense):
    rgb_shape = (1920, 1080)  # maximum resolution
    depth_shape = (1280, 720)
    device_name = 'D415'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.device is None:
            return

        # check for correct camera
        device_product_line = str(
            self.device.get_info(rs.camera_info.product_line))
        if device_product_line != 'D400':
            return

        jsonObj = json.load(
            open(os.path.join("data", "d415_config.json")))
        json_string = str(jsonObj).replace("'", '\"')
        advnc_mode = rs.rs400_advanced_mode(self.device)
        advnc_mode.load_json(json_string)

    
    def get_unique_id(self):
        if self.device is not None:
            return "realsense_" + str(self.device.get_info(rs.camera_info.serial_number))
        else:
            return "realsense_121622061798"
