import time
from models.cameras.base_camera import CamFrame, Camera
import numpy as np
import cv2
import os
import pickle
from threading import Lock

from lib.utils import load_mesh


class LogPlayer(Camera):
    def __init__(self, base_logdir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_logdir = base_logdir
        self.logs = None
        self.last_logdir = None
        self.current_frame = 0
        self.playing = False
        self.t_last_frame = time.perf_counter()
        self.log_lock = Lock()
        self.read_camera_data()

    def read_logs(self, specific_logdir):
        if specific_logdir == self.last_logdir:
            return
        self.last_logdir = specific_logdir
        logdir = os.path.join(self.base_logdir, specific_logdir)
        self.log_lock.acquire()
        self.logs = [os.path.join(logdir, x)
                     for x in os.listdir(logdir)]
        self.logs = [pickle.load(open(x, 'rb')) for x in self.logs]
        self.logs.sort(key=lambda x: x['time'])
        self.read_camera_data()
        self.log_lock.release()

    def toggle_play(self):
        self.playing = not self.playing

    def next_frame(self):
        self.current_frame += 1

    def prev_frame(self):
        self.current_frame -= 1

    def get_resolution(self) -> tuple[int, int]:
        if self.logs is None:
            return None
        return self.logs[0]['rgb'].shape[:2]

    def get_model(self):
        return load_mesh(os.path.join('data', 'models', 'logplayer.ply'))

    def get_unique_id(self):
        return "logplayer"

    def get_intrinsic_matrix(self):
        if self.logs is None:
            return None
        return self.logs[0]['intrinsic']

    def grab_frame(self) -> CamFrame:
        if self.logs is None:
            return None

        if self.log_lock.locked():
            return None

        try:
            log = self.logs[self.current_frame]
        except Exception:
            self.current_frame = 0
            log = self.logs[0]
        prev_log = self.logs[self.current_frame-1]

        playback_dt = time.perf_counter() - self.t_last_frame
        frame_dt = log['time'] - prev_log['time']
        frame_delay = frame_dt - playback_dt

        if self.playing and frame_delay < 0:
            self.t_last_frame = time.perf_counter()
            self.current_frame += 1

        self.transform(log['extrinsic'])
        output = CamFrame()
        output.depth = log['depth']
        output.rgb = log['rgb']
        output.extrinsic = log['extrinsic']
        output.intrinsic = self.intrinsic_matrix
        return output

    def close(self):
        return True
