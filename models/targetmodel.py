import copy
import time
from .model import Model
from lib.geometry import invert_homogeneous, distance_from_matrices
import numpy as np


class TargetModel(Model):
    """
    To manage models that could receive a command to be moved to a target 
    extends Model with target: Model
    """

    def __init__(self, name, mesh=None, *args, **kwargs):
        super().__init__(name, mesh, *args, **kwargs)
        self.target = Model(name=self.name+'__target', show_axes=False)
        self.linear = False

    def remove_target_vis(self, recursion=False):
        # Find absolute parent if was issued as command
        if not recursion:
            obj = self
            while obj.parent is not None:
                obj = obj.parent
            obj.remove_target_vis(recursion=True)
        else:
            if self.target is not None:
                self.target.remove()
            for child in self.children:
                if isinstance(child, TargetModel):
                    child.remove_target_vis(recursion=True)

    def transform(self, mat, incremental=False, local_frame=False):
        if self.target is not None:
            if distance_from_matrices(mat, self.target.pose) < 0.001:
                self.remove_target_vis()

        return super().transform(mat, incremental, local_frame)

    def move_to_pose(self, pose, incremental=False, local_frame=False, is_command=True, linear=False):
        """ non-blocking move to pose matrix """

        self.linear = linear

        # load target mesh at runtime
        if self.name+'__target_mesh' not in self.target.meshes.keys():
            try:
                mesh = [x for name, x in self.meshes.items()
                        if '__axes' not in name][0]
                target_mesh = copy.copy(mesh)
                target_mesh.paint_uniform_color([0, 1, 0])
                self.target.meshes[self.name +
                                   '__target__mesh'] = target_mesh
            except IndexError:
                pass

        self.target.transform(pose, incremental, local_frame)

        if is_command:
            # Find absolute parent if was issued as command
            obj = self
            self2absparent = obj.target.pose
            while obj.parent is not None:
                obj2parent = invert_homogeneous(obj.link_mat)
                self2absparent = self2absparent @ obj2parent
                obj = obj.parent
            obj.move_to_pose(self2absparent, is_command=False)
        else:
            # update targets of all children
            for child in self.children:
                if hasattr(child, 'move_to_pose'):
                    child.move_to_pose(
                        self.target.pose @ child.link_mat, is_command=False)

    def wait_for_move(self, cb=None, timeout=None) -> bool:
        """ block until target pose is reached,
            if callback specified, then move is stopped when cb returns True
            returns True if target reached, False otherwise
        """
        if timeout is not None:
            t_start = time.perf_counter()

        while True:
            if cb is not None:
                if cb():
                    return False
            if timeout is not None:
                if (time.perf_counter() - t_start) > timeout:
                    return False
            time.sleep(0.05)
            dist = distance_from_matrices(self.pose, self.target.pose, 0.2)
            if dist < 0.001:
                return True
            time.sleep(0.02)

    def stop(self):
        """ soft e-stop for robot"""
        self.move_to_pose(self.pose)  # set target to current pose

    def attach(self, obj: 'Model', matrix):
        super().attach(obj, matrix)
        self.move_to_pose(self.pose)
