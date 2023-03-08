from app.app import App
from .base_task import Task, TaskType
from lib.utils import time_stamp

import cv2
from cv2 import aruco
import math
import os


class CharBoardGenTask(Task):
    name = "Charuo Board Generation"
    description = "Generates a charuco board for the monitor defined in settings.py and saves it to data/calibration"
    type = TaskType.SETUP

    def __init__(self) -> None:
        super().__init__()

    def execute(self):
        app = self.app
        app.ar_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        sett = App().settings['Backdrop Screen Settings']

        screen_size = [x * sett['screen_diag'] /
                       math.sqrt(pow(sett['screen_ar'][0], 2) + pow(sett['screen_ar'][1], 2)) for x in sett['screen_ar']]  # in m

        pixel_size = screen_size[0] / sett['screen_res'][0]

        # old value: 28.6;  better: use multiple of pixels
        chessboardSize = 0.0286//pixel_size * pixel_size  # pixel_size * 157
        # pixel_size * 126  # old value: 0.023 # [m]
        markerSize = 0.023//pixel_size * pixel_size
        n_markers = (7, 5)

        print(f"[{time_stamp()}] Charuco Board should have squares with width: {chessboardSize} and markers with width: {markerSize}")

        app.charuco_board = aruco.CharucoBoard_create(
            n_markers[0], n_markers[1], chessboardSize, markerSize, app.ar_dict)

        width = n_markers[0] * chessboardSize  # m
        scaled_w = width / screen_size[0] * sett['screen_res'][0]

        height = n_markers[1] * chessboardSize  # m
        scaled_h = height / screen_size[1] * sett['screen_res'][1]

        imboard = app.charuco_board.draw(
            (round(scaled_w), round(scaled_h)))  # px

        hor_pad = round((sett['screen_res'][0] - scaled_w)/2)
        vert_pad = round((sett['screen_res'][1] - scaled_h)/2)
        imboard = cv2.copyMakeBorder(
            imboard, vert_pad, vert_pad, hor_pad, hor_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        my_file = os.path.join("data", "calibration", "chessboard.tiff")
        cv2.imwrite(my_file, imboard)
        return True
