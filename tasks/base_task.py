import open3d as o3d
import open3d.visualization.gui as gui
from abc import ABC, abstractmethod
from lib.utils import time_stamp
from threading import Thread
from enum import Enum, auto


class TaskType(Enum):
    SETUP = auto()
    JOB = auto()


class Task(ABC):
    """ class to manage non-periodic tasks
        for new task: - implement self.execute
                      - define self.name (human readable)
                      - optional: set self.description
    """
    name: str = None
    description: str = None
    type: TaskType

    @abstractmethod
    def execute(self):
        """ implement task logic here return True for success, or False for failure """
        pass

    def init(self, app):
        if self.name is None:
            raise AttributeError("Need to specifiy task name!")
        self.app = app

    def launch(self):
        app = self.app
        if app.task_running is not None:
            print(
                f"[{time_stamp()}] Wait for Task {app.task_running} to be finished")
            return
        app.stop_task = False
        app.task_running = self.name

        print(f"[{time_stamp()}] Task {self.name} started...")
        self.thread = Thread(target=self.run)
        self.thread.start()
        # gui.Application.instance.run_in_thread(self.run)

    def run(self):
        """ needs to be called after finishing """
        result = self.execute()
        self.app.cancel_task()
        if result:
            print(f"[{time_stamp()}] Task {self.name} finished.")
        else:
            print(f"[{time_stamp()}] Task {self.name} failed.")
