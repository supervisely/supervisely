import time


class TinyTimer:
    """Lightweight timer for measuring elapsed time since creation."""

    def __init__(self):
        self.t = time.time()

    def get_sec(self):  # since creation
        now_t = time.time()
        return now_t - self.t