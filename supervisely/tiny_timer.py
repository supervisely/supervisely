import time

from supervisely.sly_logger import logger


class TinyTimer:
    def __init__(self):
        self.t = time.time()

    def get_sec(self):  # since creation
        now_t = time.time()
        return now_t - self.t


def timeit(func):
    """Simple decorator to estimate function performance"""

    def wrapper(*args, **kwargs):
        tm = TinyTimer()
        result = func(*args, **kwargs)
        logger.debug(
            f"Function {func.__name__} executed",
            extra={"durat_msec": tm.get_sec() * 1000.0},
        )
        return result

    return wrapper
