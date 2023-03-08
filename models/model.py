import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from typing import Dict, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from app.app import App


class Model:
    parent: 'Model'
    children: List['Model']

    def __init__(self, name: str, mesh=None, show_axes=True):
        """initialization"""
        self.name = name
        self._pose = np.eye(4)
        self.meshes = {}

        self.show_axes = show_axes

        self.window = App().window
        self.scene = App().scene
        self.pov_scene = App().pov_scene

        self.active = True

        self.parent = None
        self.children = []
        self.link_mat = None
        self.is_attached = False

        if mesh is not None:
            self.meshes[self.name + '__mesh'] = mesh

        if self.show_axes:
            self.meshes[self.name +
                        "__axes"] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.add_to_scene()

    @property
    def position(self):
        return self.pose[:3, 3]

    @property
    def pose(self):
        return self._pose

    @property
    def orientation(self):
        return R.from_matrix(self.pose[:3, :3]).as_quat()

    def is_pov_mesh(self, name):
        return '__axes' not in name and '__target__mesh' not in name

    def hide(self):
        self.active = False

        def _hide():
            for name, mesh in self.meshes.items():
                self.scene.show_geometry(name, False)
                if self.is_pov_mesh(name):
                    self.pov_scene.show_geometry(name, False)

        gui.Application.instance.post_to_main_thread(self.window, _hide)

    def show(self):
        self.active = True

        def _show():
            for name, mesh in self.meshes.items():
                self.scene.show_geometry(name, True)
                if self.is_pov_mesh(name):
                    self.pov_scene.show_geometry(name, True)
        gui.Application.instance.post_to_main_thread(self.window, _show)

    def add_to_scene(self):
        gui.Application.instance.post_to_main_thread(
            self.window, self._add_to_scene)

    def _add_to_scene(self):
        standard_material = rendering.MaterialRecord()
        render_settings = App().settings["Rendering"]
        standard_material.shader = render_settings["shader"]
        standard_material.base_reflectance = render_settings['base_reflectance']
        standard_material.base_roughness = render_settings['base_roughness']

        for key, mesh in self.meshes.items():
            self.scene.add_geometry(key, mesh, standard_material)
            if self.is_pov_mesh(key):
                self.pov_scene.add_geometry(
                    key, mesh, standard_material)

    def remove(self):
        gui.Application.instance.post_to_main_thread(self.window, self._remove)

    def _remove(self):
        for name, mesh in self.meshes.items():
            self.scene.remove_geometry(name)
            if self.is_pov_mesh(name):
                self.pov_scene.remove_geometry(name)

    def _update_transform(self):
        for name, mesh in self.meshes.items():
            if not self.scene.has_geometry(name):
                self._add_to_scene()

            self.scene.set_geometry_transform(name, self.pose)
            if self.is_pov_mesh(name):
                self.pov_scene.set_geometry_transform(name, self.pose)

    def transform(self, mat, incremental=False, local_frame=False):
        """ sets the pose to the pose specified by mat
        if incremental: applies mat to current pose
        if local_frame:  appliest mat to current pose in the local frame"""
        cur_pose = self._pose
        # calculate current transformation
        if not incremental and not local_frame:
            new_pose = mat

        elif incremental and not local_frame:
            new_pose = mat @ cur_pose

        else:
            new_pose = cur_pose @ mat
        self._pose = new_pose

        if self.window is not None:
            gui.Application.instance.post_to_main_thread(
                self.window, self._update_transform)

        # update attached children
        for child in self.children:
            child.transform(self.pose @ child.link_mat)

    def attach(self, obj: 'Model', matrix):
        """ attach this object to another object, by the transformation matrix mat"""
        self.parent = obj
        self.link_mat = matrix
        self.transform(self.parent.pose @ self.link_mat)
        if self not in self.parent.children:
            self.parent.children.append(self)
        self.is_attached = True

    def unattach(self):
        self.parent.children.remove(self)
        self.parent = None
        self.link_mat = None
        self.is_attached = False
