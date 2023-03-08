import numpy as np
import multiprocessing as mp
import open3d as o3d
from lib.grasping import choose_best_grasp_pose, GraspPoseChooserConfig, GraspPoseChooserJob
from lib.utils import load_mesh
import time

from threading import Thread
from queue import Queue


class GraspPoseChooser:
    """ generates best grasp pose in separate process """

    def __init__(self) -> None:
        self.config_queue = mp.Queue(1)
        self.job_queue = mp.Queue(1)
        self.out_queue = mp.Queue(1)

        self.process = mp.Process(
            target=self.processor, args=(self.config_queue, self.job_queue, self.out_queue))

        self.process.start()

    def close(self):
        self.process.kill()
        self.process.join()

    def set_config(self, config: GraspPoseChooserConfig):
        self.config_queue.put(config, block=True)

    def get_gpose(self, job: GraspPoseChooserJob):
        self.job_queue.put(job)
        return self.out_queue.get(block=True)

    @staticmethod
    def processor(config_queue: mp.Queue, job_queue: mp.Queue, out_queue: mp.Queue):
        conf = None
        grip_dist = None
        scene = None
        while True:
            try:
                conf: GraspPoseChooserConfig = config_queue.get(block=False)
                grip_dist = conf.grip_distance
                gripper_bbox = load_mesh(
                    conf.gripper_collision_path, tensor=True)
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(gripper_bbox)
            except Exception:
                pass

            if conf is None:
                # wait for first config
                time.sleep(2.0)
                continue

            try:
                job: GraspPoseChooserJob = job_queue.get(block=False)
            except Exception:
                time.sleep(0.2)
                continue
            result = choose_best_grasp_pose(
                job, scene, grip_dist)

            out_queue.put(result)
