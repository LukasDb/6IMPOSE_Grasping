from lib.utils import time_stamp
from lib.geometry import get_6d_vector_from_affine_matrix, get_affine_matrix_from_6d_vector
from models.robot import Robot
from threading import Thread, Timer
import requests
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import re


class FanucInterface:
    ROBOT_TIMEOUT = 1.
    ROBOT_IP = "10.162.12.203"

    locked = True
    online = False
    simulate = False
    sim_fps = 20

    def __init__(self, robot: Robot, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot = robot
        self.timeout_timer = None
        self.stop_flag = False
        self.thread = Thread(target=self.run)

    def start(self):
        self.last_t = time.perf_counter()
        self.thread.start()

    def close(self):
        self.stop_flag = True
        self.thread.join()
        self.thread_vis.join()

    def set_simulate(self, state):
        self.simulate = state

    def set_lock(self, state):
        self.locked = state

    def update_ip(self, new_ip):
        self.ROBOT_IP = new_ip

    def run(self):
        while not self.stop_flag:
            if self.locked:  # if locked dont do anything
                time.sleep(0.2)
                continue

            if not self.simulate:
                if not self.online:
                    current_pose = self.reset()
                else:
                    current_pose = self.send_and_receive()
                if current_pose is None:
                    #print("Exception: ", e)
                    self.online = False
                    time.sleep(0.5)
                    continue
                self.online = True

            else:
                current_pose = self.step_simulation()
                self.online = False

            # move model/visualization to reported robot pose
            self.robot.endeffector.transform(current_pose)

            if self.timeout_timer is not None:
                # reset timeout timer
                self.timeout_timer.cancel()

            self.timeout_timer = Timer(
                self.ROBOT_TIMEOUT, self.robot_timeout)
            self.timeout_timer.start()

        # after loop is stopped, cancel timer
        if self.timeout_timer is not None:
            self.timeout_timer.cancel()

    def step_simulation(self):
        dt = time.perf_counter() - self.last_t
        if dt < 1/self.sim_fps:
            time.sleep(1/self.sim_fps - dt)
        dt = time.perf_counter() - self.last_t
        self.last_t = time.perf_counter()

        pos_step = 0.3 / self.sim_fps  # for simulation
        rot_step = 100 / 180 * np.pi / self.sim_fps
        width_step = 0.1 / self.sim_fps

        current_pose = self.robot.endeffector.pose

        # update position
        vec_pos = self.robot.endeffector.target.pose[:3,
                                                     3] - self.robot.endeffector.pose[:3, 3]
        pos_dist = np.linalg.norm(vec_pos)
        d_pos = np.clip(pos_dist, -pos_step, pos_step)
        vec_d_pos = vec_pos / pos_dist * d_pos if pos_dist > 0.00001 else vec_pos
        current_pose[:3, 3] += vec_d_pos

        # update rotation
        d_rot = R.from_matrix(self.robot.endeffector.target.pose[:3, :3]) * \
            R.from_matrix(self.robot.endeffector.pose[:3, :3]).inv()
        if d_rot.magnitude() / rot_step > 1.0:
            key_rots = R.concatenate([R.from_matrix(self.robot.endeffector.pose[:3, :3]), R.from_matrix(
                self.robot.endeffector.target.pose[:3, :3])])
            key_times = [0, d_rot.magnitude() / rot_step]
            slerp = Slerp(key_times, key_rots)
            interp_rot = slerp(1).as_matrix()
            current_pose[:3, :3] = interp_rot
        else:
            current_pose[:3, :3] = self.robot.endeffector.target.pose[:3, :3]

        # update gripper
        d_width = self.robot.gripper.target_width - self.robot.gripper.gripper_width
        d_width = np.clip(d_width, -width_step, width_step)
        self.robot.gripper.update_gripper(
            self.robot.gripper.gripper_width + d_width)
        return current_pose

    def send_and_receive(self):
        robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
        url = robot_http + "remotecontrol"

        pose6d = self.__mat_to_fanuc_6d(self.robot.endeffector.target.pose)

        http_params = {
            'x': pose6d[0],
            'y': pose6d[1],
            'z': pose6d[2],
            'w': pose6d[3],
            'p': pose6d[4],
            'r': pose6d[5],
            'linear_path': 1 if self.robot.gripper.linear else 0,
            'interrupt': 0,
            'gripper_width': (self.robot.gripper.target_width)*1000,  # to mm
            'gripper_force': self.robot.gripper.target_force,
        }
        try:
            req = requests.get(url, params=http_params, timeout=1.0)
            jdict = json.loads(req.text)
            new_pose, joint_positions = self.__parse_remote_position(
                jdict['remote_position'])
            gripper_width = self.__parse_gripper_offset(
                jdict['remote_gripper_info'])
        except (KeyError, json.decoder.JSONDecodeError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            return None

        # update robot visualization
        self.robot.update_robot_visualization(joint_positions)

        # update link between robot and gripper
        self.robot.gripper.update_gripper(gripper_width)

        return new_pose

    def __parse_remote_position(self, result):
        pose = np.array([result['x'], result['y'], result['z'],
                         result['w'], result['p'], result['r']])
        joint_positions = [result["j1"], result["j2"],
                           result["j3"], result["j4"], result["j5"], result["j6"]]
        return self.__fanuc_6d_to_mat(pose), joint_positions

    def __parse_gripper_offset(self, result):
        return result['width']/1000

    def reset(self):
        """ move target to current pose"""

        if not self.simulate:
            # manually get robot pose, update children and then set target to current pose
            try:
                robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
                url = robot_http + "remoteposition"
                req = requests.get(url, timeout=1.0)
                jdict = json.loads(req.text)
                current_pose, joint_positions = self.__parse_remote_position(
                    jdict)
                self.robot.update_robot_visualization(joint_positions)
            except (KeyError, json.decoder.JSONDecodeError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                return None
        else:
            current_pose = self.robot.endeffector.target.pose

        self.robot.endeffector.transform(current_pose)

        for child in self.robot.endeffector.children:
            if hasattr(child, 'move_to_pose'):
                child.move_to_pose(child.pose)

        return current_pose

    def robot_timeout(self):
        self.online = False
        print(f"[{time_stamp()}] Robot timed out.")

    def __fanuc_6d_to_mat(self, vec_6d):
        vec_6d[:3] = vec_6d[:3] / 1000.
        vec_6d[3:] = vec_6d[3:] / 180 * np.pi

        new_pose = get_affine_matrix_from_6d_vector(
            'xyz', vec_6d)  # @ gripper_offset
        return new_pose

    def __mat_to_fanuc_6d(self, mat):
        pose6d = get_6d_vector_from_affine_matrix('xyz', mat)
        pose6d[:3] = pose6d[:3] * 1000
        pose6d[3:] = pose6d[3:] / np.pi * 180
        return pose6d
