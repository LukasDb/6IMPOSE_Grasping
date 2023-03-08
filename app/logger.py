import multiprocessing as mp
import pickle
import os
from datetime import datetime


class Logger:
    log_writer: mp.Process
    log_queue: mp.Queue

    def __init__(self) -> None:
        self.log_queue = mp.Queue()
        self.log_writer = mp.Process(
            target=self.log_process, args=(self.log_queue, ))

        self.log_writer.start()

    def close(self):
        self.log_writer.kill()
        self.log_writer.join()

    def log(self, data, logdir):
        self.log_queue.put((data, logdir))

    @staticmethod
    def log_process(queue: mp.Queue):

        while True:
            data, logdir = queue.get()
            time_now = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            logname = f"log_{time_now}"
            try:
                os.makedirs(logdir)
            except FileExistsError:
                pass

            pickle.dump(data, open(os.path.join(logdir, logname), 'wb'))
