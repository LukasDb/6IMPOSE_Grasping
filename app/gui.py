import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import cv2
from scipy.spatial.transform import Rotation as R

from app.app import write_settings
from app.xyz_controller import XYZController
from app.digital_twin import DigitalTwin
from app.app import App
from tasks.base_task import TaskType
from models.cameras import CamFrame
from threading import Thread
from typing import Dict
import time
import numpy as np
import os


class GUI(XYZController):
    MENU_QUIT = 3
    MENU_ABOUT = 21

    def __init__(self):
        super().__init__()
        self.previews = {}
        self.task_buttons = []

        self.window = w = gui.Application.instance.create_window(
            "Human Robot Interaction", 1820, 880, 0, 0)

        # --- 3D visualization ---
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.look_at([0., 0.2, 0., ], [1., 1., 1.], [0., 0., 1.])
        self._scene.scene.show_ground_plane(
            True, rendering.Scene.GroundPlane.XY)

        self._pov_scene = gui.SceneWidget()
        self._pov_scene.scene = rendering.Open3DScene(w.renderer)
        self._pov_scene.look_at([0., 0., 0., ], [2., 2., 2.], [0., 0., 1.])
        self._pov_scene.scene.show_axes(False)

        App().window = self.window
        App().scene = self._scene.scene
        App().pov_scene = self._pov_scene.scene
        
        self.app = DigitalTwin(cb_camera=self.cb_camera)
        w.set_on_close(self.app.close)

        # build GUI
        self.em = em = w.theme.font_size
        self.margins = gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)

        self.build_task_buttons()
        self.status_panel = self.build_status_panel()
        self.settings_panel = self.build_settings_panel()
        self.previews_frame = self.build_previews()

        self.update_ui_status()

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._pov_scene)
        w.add_child(self.settings_panel)
        w.add_child(self.status_panel)
        w.add_child(self.previews_frame)
        self.create_menu()

        mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE
        self._scene.set_view_controls(mouse_model)
        self._pov_scene.set_on_mouse(
            lambda event: gui.SceneWidget.EventCallbackResult.CONSUMED)

        self.apply_render_settings()

        self._select_dt_object.selected_index = 0   # select first object in GUI
        self._select_camera.selected_index = 0      # select first object in GUI

        self.app.threads.append(Thread(target=self.update_status_bar_thread))
        self.app.threads.append(Thread(target=self.check_dirty))

        self.t_last_cb_camera = time.perf_counter()
        self.app.start()

    def check_dirty(self):
        while not self.app.stop_flag:
            time.sleep(1)
            if self.app.dirty:
                self.update_ui_status()
                self.app.dirty = False

    def update_ui_status(self):
        """ to be called when things have changed """
        self._select_dt_object.clear_items()
        self._select_camera.clear_items()
        self._select_log.clear_items()
        for item in self.app.dt_objectcs_list.keys():
            self._select_dt_object.add_item(item)
        for item in self.app.camera_list.keys():
            self._select_camera.add_item(item)
        for item in self.app.log_list:
            self._select_log.add_item(item)

    def cb_camera(self, frame: CamFrame):
        # update previews in main thread
        depth_colormap = cv2.cvtColor(
            cv2.applyColorMap(
                cv2.convertScaleAbs(frame.depth, alpha=255 / 1.),
                cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)
        previews = {"RGB": frame.rgb,
                    "Depth": depth_colormap, "Darknet": frame.rgb2}
        self.update_previews(previews)

        try:
            img_frame = list(self.previews.values())[0].frame
        except Exception:
            return

        if img_frame.width < 1e-6:
            return

        pose = frame.extrinsic
        #pose = self.app.camera_list[self.app.EYE_IN_HAND].pose

        pos = pose[:3, 3]
        up = -pose[:3, 1]  # -y axis
        at = np.array([0, 0, 1., 1.]) @ pose.T  # +z axis

        h, w = frame.rgb.shape[:2]
        real_img_ar = h/w
        align_fov = frame.intrinsic.copy()
        current_ar = img_frame.height / img_frame.width
        scale = min(real_img_ar / current_ar, 1)
        align_fov[0, 0] *= scale
        align_fov[1, 1] *= scale
        self._pov_scene.scene.camera.set_projection(
            align_fov, 0.01, 3.0, w, h)
        self._pov_scene.look_at(at[:3], pos, up)

    def update_previews(self, previews: Dict):
        def __update():
            for name, img in previews.items():
                if img is None:
                    continue
                if name not in self.previews:
                    preview = gui.ImageWidget()
                    self.previews[name] = preview
                    self.previews[name].update_image(
                        o3d.geometry.Image(img))
                    self.previews_frame.add_tab(name, preview)
                else:
                    self.previews[name].update_image(
                        o3d.geometry.Image(img))
        gui.Application.instance.post_to_main_thread(self.window, __update)

    def update_status_bar_thread(self):
        while not self.app.stop_flag:
            time.sleep(0.2)
            gui.Application.instance.post_to_main_thread(
                self.window, self._update_status_bar)

    def apply_render_settings(self):
        # show global frame
        self._scene.scene.show_axes(App().settings['Rendering']['show_axes'])

        # lighting intensity
        intensity = App().settings['Rendering']['ibl_intensity']
        self._scene.scene.scene.set_indirect_light_intensity(intensity)
        self._pov_scene.scene.scene.set_indirect_light_intensity(intensity)

        standard_material = rendering.MaterialRecord()
        render_settings = App().settings["Rendering"]
        standard_material.shader = render_settings["shader"]
        standard_material.base_reflectance = render_settings['base_reflectance']
        standard_material.base_roughness = render_settings['base_roughness']
        self._scene.scene.update_material(standard_material)
        self._pov_scene.scene.update_material(standard_material)

    def build_settings_panel(self):
        controls = self.build_controls()
        setup = self.build_setup()
        settings = self.build_settings()
        self.build_task_interface()

        # more right spacing
        margins = gui.Margins(left=0.25*self.em, right=1.5*self.em,
                              top=0.25*self.em, bottom=1.5*self.em)

        settings_panel = gui.ScrollableVert(
            0.25*self.em, margins)

        settings_panel.add_child(setup)
        settings_panel.add_child(settings)
        settings_panel.add_child(controls)
        settings_panel.add_child(self.task_iface)
        return settings_panel

    def build_status_panel(self):
        status_panel = gui.Horiz(
            0.5*self.em, self.margins)
        self.robot_status = gui.Label("----------------------------------")
        self.task_status = gui.Label("----------------------------------")
        status_panel.add_child(self.robot_status)
        status_panel.add_child(self.task_status)
        return status_panel

    def build_task_buttons(self):
        self.jobs_col = gui.Vert(0.25 * self.em, self.margins)
        self.setups_col = gui.Vert(0.25 * self.em, self.margins)
        for task in self.app.tasks:
            btn = gui.Button(f"{task.name}")
            btn.toggleable = False
            if task.description is not None:
                btn.tooltip = task.description
            btn.set_on_clicked(task.launch)
            self.task_buttons.append(btn)
            if task.type == TaskType.JOB:
                self.jobs_col.add_child(btn)
            elif task.type == TaskType.SETUP:
                self.setups_col.add_child(btn)
            else:
                raise AssertionError(
                    f"Unknown task type {task.type} of {task}")

    def build_previews(self):
        # previews_frame = gui.CollapsableVert("Previews", 0.25 * self.em,
        #                                     gui.Margins(self.em, 0, 0, 0))
        previews_frame = gui.TabControl()
        return previews_frame

    def build_setup(self):
        em = self.em
        setup = gui.CollapsableVert("Setup", 0.25 * em,
                                    self.margins)
        setup.set_is_open(False)
        setup.add_child(self.setups_col)
        return setup

    def build_settings(self):
        em = self.em
        self.set = None

        def update_settings(val):
            """ map GUI settings to settings structs"""
            def update_rec(gui_set, set):
                for child in gui_set.get_children():
                    if isinstance(child, gui.Horiz):
                        # is a settings entry
                        # setattr(set, child.name, child.value)
                        gui_name, _, gui_val = child.get_children()
                        if gui_val.tooltip == 'float':
                            val = gui_val.double_value
                        elif gui_val.tooltip == 'int':
                            val = int(gui_val.int_value)
                        elif gui_val.tooltip == 'array':
                            val = np.array(eval(gui_val.text_value))
                        elif gui_val.tooltip == 'bool':
                            val = gui_val.checked
                        elif gui_val.tooltip == 'string':
                            val = gui_val.text_value

                        set[gui_name.text] = val
                    elif isinstance(child, gui.CollapsableVert):
                        # is a settings container
                        update_rec(child, set[child.tooltip])

            update_rec(self.set, App().settings)
            self.apply_render_settings()

        def construct_setting(settings_dict, set_name, root=False):
            set = gui.CollapsableVert(set_name, 0.25 * em,
                                      gui.Margins(em, 0, 0, 0))
            set.tooltip = set_name

            if root:
                btn = gui.Button('Save Settings')
                btn.toggleable = False
                btn.set_on_clicked(write_settings)
                set.add_child(btn)

            set.set_is_open(False)
            for name, val in settings_dict.items():
                # print(f"Found {name}: {val}")
                entry = None
                if isinstance(val, dict):
                    entry = construct_setting(val, name)
                elif isinstance(val, float):
                    entry = gui.Horiz()
                    entry.add_child(gui.Label(name))
                    number_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                    number_edit.double_value = val
                    number_edit.set_on_value_changed(update_settings)
                    number_edit.tooltip = 'float'
                    entry.add_stretch()
                    entry.add_child(number_edit)
                elif type(val) == int:
                    entry = gui.Horiz()
                    entry.add_child(gui.Label(name))
                    number_edit = gui.NumberEdit(gui.NumberEdit.INT)
                    number_edit.int_value = val
                    number_edit.set_on_value_changed(update_settings)
                    number_edit.tooltip = 'int'
                    entry.add_stretch()
                    entry.add_child(number_edit)
                elif isinstance(val, list):
                    entry = gui.Horiz()
                    entry.add_child(gui.Label(name))
                    edit = gui.TextEdit()
                    edit.text_value = str(val)
                    edit.set_on_value_changed(update_settings)
                    edit.tooltip = 'array'
                    entry.add_stretch()
                    entry.add_child(edit)
                elif type(val) == bool:
                    entry = gui.Horiz()
                    entry.add_child(gui.Label(name))
                    edit = gui.Checkbox('')
                    edit.checked = val
                    edit.set_on_checked(update_settings)
                    edit.tooltip = 'bool'
                    entry.add_stretch()
                    entry.add_child(edit)
                elif type(val) == str:
                    entry = gui.Horiz()
                    entry.add_child(gui.Label(name))
                    edit = gui.TextEdit()
                    edit.text_value = str(val)
                    edit.set_on_value_changed(update_settings)
                    edit.tooltip = 'string'
                    entry.add_stretch()
                    entry.add_child(edit)

                if entry is not None:
                    set.add_child(entry)

            return set

        self.set = set = construct_setting(
            App().settings, 'Settings', root=True)

        return set

    def build_task_interface(self):
        self._cancel_task_btn = gui.Button("Cancel Task")
        self._cancel_task_btn.toggleable = False
        self._cancel_task_btn.enabled = True
        self._cancel_task_btn.set_on_clicked(self.cancel_task)
        self.jobs_col.add_fixed(self.em)
        self.jobs_col.add_child(self._cancel_task_btn)

        self.task_iface = gui.CollapsableVert(
            "Tasks", 0.25 * self.em, self.margins)
        self.task_iface.add_child(self.jobs_col)

    def build_controls(self):
        em = self.em
        controls = gui.CollapsableVert("Controls", 0.25 * em, self.margins)

        def _dt_object_selected(selected, index):
            self.app.selected_dt_object = selected

        dt_select_container = gui.Horiz(0.25*em, self.margins)
        self._select_dt_object = gui.Combobox()
        self._select_dt_object.set_on_selection_changed(_dt_object_selected)
        dt_select_container.add_child(gui.Label("Object: "))
        dt_select_container.add_stretch()
        dt_select_container.add_child(self._select_dt_object)
        controls.add_child(dt_select_container)

        def _camera_selected(selected, index):
            self.app.selected_camera = selected

        cam_select_container = gui.Horiz(0.25*em, self.margins)
        self._select_camera = gui.Combobox()
        self._select_camera.set_on_selection_changed(_camera_selected)
        cam_select_container.add_child(gui.Label("Camera: "))
        cam_select_container.add_stretch()
        cam_select_container.add_child(self._select_camera)
        controls.add_child(cam_select_container)

        def _log_selected(selected_log, index):
            self.app.selected_log = selected_log

        log_select_container = gui.Horiz(0.25*em, self.margins)
        self._select_log = gui.Combobox()
        self._select_log.selected_text = " - "
        self._select_log.set_on_selection_changed(_log_selected)
        log_select_container.add_child(gui.Label("Log folder: "))
        log_select_container.add_stretch()
        log_select_container.add_child(self._select_log)
        controls.add_child(log_select_container)

        self._lock_robot = gui.ToggleSwitch("Lock Robot")
        self._lock_robot.set_on_clicked(self.app.robot.set_lock)
        self._lock_robot.is_on = self.app.lock_robot
        controls.add_child(self._lock_robot)

        self._simulate_robot = gui.ToggleSwitch("Simulate Robot")
        self._simulate_robot.set_on_clicked(
            self.app.robot.set_simulate)
        self._simulate_robot.is_on = self.app.simulate_robot
        controls.add_child(self._simulate_robot)

        self._freeze_object = gui.ToggleSwitch("Freeze Object")
        self._freeze_object.set_on_clicked(self._on_freeze_object)
        self._freeze_object.is_on = self.app.freeze_object
        controls.add_child(self._freeze_object)

        self._draw_object = gui.ToggleSwitch("Draw Object")
        self._draw_object.set_on_clicked(self._on_draw_object)
        self._draw_object.is_on = self.app.draw_object
        controls.add_child(self._draw_object)

        self._use_icp = gui.ToggleSwitch("Use ICP")
        self._use_icp.set_on_clicked(self._on_use_icp)
        self._use_icp.is_on = self.app.use_icp
        controls.add_child(self._use_icp)

        self._enable_log = gui.ToggleSwitch("Log Data")
        self._enable_log.set_on_clicked(self._on_enable_log)
        self._enable_log.is_on = self.app.log
        self._enable_log.tooltip = "Continuously record every camera frame"
        controls.add_child(self._enable_log)

        self.log_controls = gui.Horiz(
            em, self.margins)

        self._single_log = gui.Button("Write")
        self._single_log.tooltip = "Record a single camera frame"
        self._single_log.toggleable = False
        self._single_log.enabled = True
        self._single_log.set_on_clicked(self.set_single_log)

        self._play_log = gui.Button("Play")
        self._play_log.toggleable = True
        self._play_log.set_on_clicked(self.app.logplayer.toggle_play)
        self._play_log.is_on = self.app.play_log

        self._prev_log = gui.Button("-")
        self._prev_log.toggleable = False
        self._prev_log.enabled = True
        self._prev_log.set_on_clicked(self.app.logplayer.prev_frame)

        self._next_log = gui.Button("+")
        self._next_log.toggleable = False
        self._next_log.enabled = True
        self._next_log.set_on_clicked(self.app.logplayer.next_frame)

        self.log_controls.add_child(self._prev_log)
        self.log_controls.add_child(self._play_log)
        self.log_controls.add_child(self._next_log)
        self.log_controls.add_stretch()
        self.log_controls.add_child(self._single_log)

        controls.add_child(self.log_controls)

        self._reset_target = gui.Button("Reset Target")
        self._reset_target.toggleable = False
        self._reset_target.enabled = True
        self._reset_target.set_on_clicked(self.app.robot.reset)
        controls.add_child(self._reset_target)

        def move_to_log():
            cam = self.app.camera_list[self.app.EYE_IN_HAND]
            cam.move_to_pose(self.app.logplayer.pose)
        self._move_log = gui.Button("Move to logplayer")
        self._move_log.toggleable = False
        self._move_log.enabled = True
        self._move_log.set_on_clicked(move_to_log)
        controls.add_child(self._move_log)

        # for task_button, task in zip(self.task_buttons, self.app.tasks):
        #    def launch():
        # self._cancel_task_btn.enabled = True
        # for btn in self.task_buttons:
        #    btn.enabled = False
        #        task.launch()
        #    task_button.set_on_clicked(launch)
        robot_controller = self.generate_XYZController(
            self.app.control_robot, self.em, self.margins)
        controls.add_child(robot_controller)

        return controls

    def _update_status_bar(self):
        robot_status = self.app.robot.online
        # robot status
        if robot_status:
            pos = 1000 * self.app.robot_arm.position
            rot = R.from_matrix(
                self.app.robot_arm.pose[:3, :3]).as_euler('xyz', degrees=True)
            #
            # : [{:7.1f}, {:7.1f}, {:7.1f}] mm,  [{:6.1f}, {:6.1f}, {:6.1f}]°".format(*pos, *rot)
            robot_status_text = "Online"

        else:
            robot_status_text = "Offline"

        # task status
        if self.app.task_running is None:
            task_status_text = "No Task running.                                                                 "
        else:
            task_status_text = "Active Task: " + self.app.task_running

        # update gui/print
        self.robot_status.text = robot_status_text
        self.task_status.text = task_status_text
        self.robot_status.text_color = gui.Color(
            r=0, g=1.0, b=0) if robot_status else gui.Color(r=1.0, g=0, b=0)

    def create_menu(self):
        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", self.MENU_QUIT)
            help_menu = gui.Menu()
            help_menu.add_item("About", self.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        w = self.window
        w.set_on_menu_item_activated(self.MENU_QUIT, self.app.close)
        w.set_on_menu_item_activated(self.MENU_ABOUT, self._on_menu_about)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        width = 20 * layout_context.theme.font_size
        # Rect(x_min, y_min, width, height) # top left is (x_min, y_min)

        status_panel_height = self.status_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        self.settings_panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, r.height)

        self.status_panel.frame = gui.Rect(r.x, r.get_bottom()-status_panel_height, r.width-width,
                                           status_panel_height)

        spacing = 0.25*self.em
        scene_x = (r.width-width)//2 - spacing
        scene_y = (r.height-status_panel_height - 2*spacing) // 2

        # build middle setup
        self._scene.frame = gui.Rect(
            r.x+spacing+scene_x, r.y, scene_x, 2*scene_y + spacing)
        self._pov_scene.frame = gui.Rect(
            r.x, r.y, scene_x, scene_y)
        self.previews_frame.frame = gui.Rect(
            r.x, r.y+scene_y+spacing, scene_x, scene_y)

        # change width of labels in status bar
        init = self.robot_status.frame
        status_panel_width = self.status_panel.frame.width
        robot_width = status_panel_width // 3
        task_width = robot_width*2

        self.robot_status.frame = gui.Rect(
            init.x, init.y, robot_width, init.height)
        self.task_status.frame = gui.Rect(
            init.x+robot_width, init.y, task_width, init.height)

    def cancel_task(self):
        # activate all launch buttons
        # for _task in self.tasks:
        #    _task.btn.enabled = True
        # self._cancel_task_btn.enabled = False
        self.app.cancel_task()

    def _on_freeze_object(self, state):
        self.app.freeze_object = state

    def _on_draw_object(self, state):
        self.app.draw_object = state

    def _on_use_icp(self, state):
        self.app.use_icp = state

    def _on_enable_log(self, state):
        self.app.log = state

    def set_single_log(self):
        self.app.single_log = True

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(
            "Visualizer for Human Robot Interaction\nAuthor: Lukas Dirnberger\nlukas.dirnberger@tum.de"))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()
