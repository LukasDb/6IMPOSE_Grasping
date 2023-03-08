from math import pi
import numpy as np
import open3d
import open3d.visualization.gui as gui


class XYZController:
    step = 0.05
    step_r = 5/180*pi

    def update_step(self, value):
        self.step = value

    def update_rot(self, value):
        self.step_r = value / 180. * pi

    def generate_XYZController(self, robot_control, em, margins):
        """ a simple grid with pushbuttons for x, y, z with + and - for each one """
        inc_x = gui.Button('X+')
        dec_x = gui.Button('X-')
        inc_y = gui.Button('Y+')
        dec_y = gui.Button('Y-')
        inc_z = gui.Button('Z+')
        dec_z = gui.Button('Z-')

        inc_r = gui.Button('P+')  # naming in the code is wrong!
        dec_r = gui.Button('P-')
        inc_p = gui.Button('Y+')
        dec_p = gui.Button('Y-')
        inc_w = gui.Button('R+')
        dec_w = gui.Button('R-')
        inc_width = gui.Button('Width+')
        dec_width = gui.Button('Width-')
        open = gui.Button('Open')
        close = gui.Button('Close')

        def inc_x_fnc():
            robot_control([self.step, 0, 0, 0, 0, 0, 0, 0])

        def inc_y_fnc():
            robot_control([0, self.step, 0, 0, 0, 0, 0, 0])

        def inc_z_fnc():
            robot_control([0, 0, self.step, 0, 0, 0, 0, 0])

        def dec_x_fnc():
            robot_control([-self.step, 0, 0, 0, 0, 0, 0, 0])

        def dec_y_fnc():
            robot_control([0, -self.step, 0, 0, 0, 0, 0, 0])

        def dec_z_fnc():
            robot_control([0, 0, -self.step, 0, 0, 0, 0, 0])

        def inc_r_fnc():
            robot_control([0, 0, 0, self.step_r, 0, 0, 0, 0])

        def inc_p_fnc():
            robot_control([0, 0, 0, 0, self.step_r, 0, 0, 0])

        def inc_w_fnc():
            robot_control([0, 0, 0, 0, 0, self.step_r, 0, 0])

        def dec_r_fnc():
            robot_control([0, 0, 0, -self.step_r, 0, 0, 0, 0])

        def dec_p_fnc():
            robot_control([0, 0, 0, 0, -self.step_r, 0, 0, 0])

        def dec_w_fnc():
            robot_control([0, 0, 0, 0, 0, -self.step_r, 0, 0])

        def inc_width_fnc():
            robot_control([0, 0, 0, 0, 0, 0, 0, self.step])

        def dec_width_fnc():
            robot_control([0, 0, 0, 0, 0, 0, 0, -self.step])

        def open_gripper():
            robot_control([0, 0, 0, 0, 0, 0, 0, 1])

        def close_gripper():
            robot_control([0, 0, 0, 0, 0, 0, 0, -1])

        inc_x.set_on_clicked(inc_x_fnc)
        inc_y.set_on_clicked(inc_y_fnc)
        inc_z.set_on_clicked(inc_z_fnc)
        dec_x.set_on_clicked(dec_x_fnc)
        dec_y.set_on_clicked(dec_y_fnc)
        dec_z.set_on_clicked(dec_z_fnc)

        inc_r.set_on_clicked(inc_r_fnc)
        inc_p.set_on_clicked(inc_p_fnc)
        inc_w.set_on_clicked(inc_w_fnc)
        dec_r.set_on_clicked(dec_r_fnc)
        dec_p.set_on_clicked(dec_p_fnc)
        dec_w.set_on_clicked(dec_w_fnc)

        inc_width.set_on_clicked(inc_width_fnc)
        open.set_on_clicked(open_gripper)
        dec_width.set_on_clicked(dec_width_fnc)
        close.set_on_clicked(close_gripper)

        grid_pos = gui.VGrid(cols=2, spacing=0.25*em, margins=margins)
        grid_rot = gui.VGrid(cols=2, spacing=0.25*em, margins=margins)
        grid_gripper = gui.VGrid(cols=2, spacing=0.25*em, margins=margins)

        grid_pos.add_child(inc_x)
        grid_pos.add_child(dec_x)
        grid_pos.add_child(inc_y)
        grid_pos.add_child(dec_y)
        grid_pos.add_child(inc_z)
        grid_pos.add_child(dec_z)

        grid_rot.add_child(inc_r)
        grid_rot.add_child(dec_r)
        grid_rot.add_child(inc_p)
        grid_rot.add_child(dec_p)
        grid_rot.add_child(inc_w)
        grid_rot.add_child(dec_w)

        grid_gripper.add_child(open)
        grid_gripper.add_child(close)
        grid_gripper.add_child(inc_width)
        grid_gripper.add_child(dec_width)

        pos_rot = gui.Horiz(0.25*em, margins)
        pos_rot.add_stretch()
        pos_rot.add_child(grid_pos)
        pos_rot.add_stretch()
        pos_rot.add_child(grid_rot)
        pos_rot.add_stretch()

        gripper = gui.Horiz(0.25*em, margins)
        gripper.add_stretch()
        gripper.add_child(grid_gripper)
        gripper.add_stretch()

        grid = gui.Vert(0.25*em, margins)
        grid.add_child(pos_rot)
        grid.add_child(gripper)

        xyz_controller = gui.Vert(0.25*em, margins)

        step_container = gui.Horiz(0.25*em, margins)
        _step = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        _step.double_value = self.step
        _step.tooltip = 'Set move step'
        _step.set_on_value_changed(self.update_step)
        step_container.add_child(gui.Label('Step'))
        step_container.add_stretch()
        step_container.add_child(_step)

        rot_container = gui.Horiz(0.25*em, margins)
        _rot = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        _rot.double_value = self.step_r/pi*180
        _rot.tooltip = 'Set rotation step'
        _rot.set_on_value_changed(self.update_rot)
        rot_container.add_child(gui.Label('Rot'))
        rot_container.add_stretch()
        rot_container.add_child(_rot)

        xyz_controller.add_child(step_container)
        xyz_controller.add_child(rot_container)
        xyz_controller.add_child(grid)

        return xyz_controller
