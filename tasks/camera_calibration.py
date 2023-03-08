from app.app import App
from .base_task import Task, TaskType
from .charboard_generation import CharBoardGenTask
import open3d as o3d
import open3d.visualization.gui as gui
from lib.utils import time_stamp


class CameraCalibrationTask(Task):
    name = "Camera Calibration"
    description = "Calibrate the flange matrix for eye-in-hand cameras with present robot poses and images for this camera."
    type = TaskType.SETUP

    _attach_to_robot = False
    selected_camera = None

    def execute(self):
        app = self.app
        dlg = gui.Dialog("Camera Calibration")
        # Add the text
        em = App().window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        camera_listview = gui.ListView()
        camera_listview.set_items(list(app.camera_list.keys()))
        camera_listview.set_on_selection_changed(self._on_camera_selected)

        attach_to_robot = gui.Checkbox("Attach to robot")
        attach_to_robot.set_on_checked(self._on_attach_to_robot)
        attach_to_robot.checked = self._attach_to_robot

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_calibrate_camera)
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(self._on_dialog_close)
        buttons = gui.Horiz(em, gui.Margins(em, em, em, em))
        buttons.add_child(ok)
        buttons.add_child(cancel)

        dlg_layout.add_child(camera_listview)
        dlg_layout.add_child(attach_to_robot)
        dlg_layout.add_child(buttons)

        dlg.add_child(dlg_layout)
        App().window.show_dialog(dlg)

        return True

    def _on_dialog_close(self):
        App().window.close_dialog()

    def _on_camera_selected(self, new_cam, is_double_click):
        self.selected_camera = new_cam
        if is_double_click:
            self._on_calibrate_camera()

    def _on_attach_to_robot(self, state):
        self._attach_to_robot = state

    def _on_calibrate_camera(self):
        App().window.close_dialog()
        app = self.app

        print(f"[{time_stamp()}] Calibrating {self.selected_camera}...")
        cam = app.camera_list[self.selected_camera]

        # launch charuco generation if necessary
        if app.ar_dict is None or app.charuco_board is None:
            print(f"[{time_stamp()}] Need to create charuco board first!")
            return

        flange_matrix = cam.calibrate_eyeInHand(app.ar_dict, app.charuco_board)

        print(f"[{time_stamp()}] Found extrinsic matrix:\n{flange_matrix}\n")

        if self._attach_to_robot:
            cam.attach(app.robot_arm, flange_matrix)
