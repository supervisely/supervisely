import functools
import time
from supervisely_lib.sly_logger import logger


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        logger.debug(f"TIME {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer