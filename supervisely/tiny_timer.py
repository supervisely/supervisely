import time

from supervisely.sly_logger import logger


class TinyTimer:
    def __init__(self):
        self.t = time.time()

    def get_sec(self):  # since creation
        now_t = time.time()
        return now_t - self.t
