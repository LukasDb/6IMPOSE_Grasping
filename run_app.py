import begin
import os
import sys
import multiprocessing as mp
sys.path.append(os.path.join("networks", "pvn"))
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


@begin.start
def run(task=None, gui=True):

    import silence_tensorflow.auto
    import tensorflow as tf

    import multiprocessing as mp
    mp.set_start_method('spawn')

    physical_devices = tf.config.list_physical_devices('GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    tf.config.optimizer.set_experimental_options({"debug_stripper": True})

    if gui and task is None:
        # from app import HRIClient
        from app.gui import GUI
        import open3d.visualization.gui as gui

        gui.Application.instance.initialize()
        window = GUI()
        gui.Application.instance.run()
    else:
        import time
        from app.digital_twin import DigitalTwin

        def cb_camera(frame):
            fps = 1 / (time.perf_counter() - cb_camera.last_t)
            fps = 0.5*fps + 0.5 * cb_camera.last_FPS
            cb_camera.last_FPS = fps
            print(f"FPS: {fps}")
            cb_camera.last_t = time.perf_counter()

        cb_camera.last_t = time.perf_counter()
        cb_camera.last_FPS = 15

        #app = DigitalTwin(cb_camera)
        app = DigitalTwin()
        print("init done")
        # select cam and object
        app.selected_camera = 'logplayer'
        app.selected_dt_object = 'cpsduck__00'
        app.start()
        time.sleep(10)
        print("running?")
        # setup simulated robot
        app.robot_interface.set_simulate(True)
        app.robot_interface.set_lock(False)

        if task == 'test':
            # move robot to logplayer
            cam = app.camera_list[app.EYE_IN_HAND]
            cam.move_to_pose(app.logplayer.pose)
            cam.wait_for_move()
            t_start = time.perf_counter()
            while (time.perf_counter() - t_start) < 10:
                time.sleep(1)
                print()
                print(list(app.dt_objectcs_list.values())[0].position)
        elif task is not None:
            try:
                selected_task = app.tasks[[
                    t.name for t in app.tasks].index(task)]
            except ValueError:
                print(
                    f"Selected task \"{task}\" is not available from tasks {[x.name for x in app.tasks]}")
                exit(-1)
            selected_task.launch()
            t_start = time.perf_counter()
            while app.task_running is not None:
                time.sleep(0.2)
                print()
                print(
                    f"Pos: {app.robot_arm.position}, Gripper: {app.gripper.gripper_width}")
        app.close()
