import concurrent.futures
import threading
import time
from supervisely import logger
from .request_queue import RequestQueue


class TrainingStoppedException(Exception):
    pass


class BackgroundRequestHandler:
    def __init__(self, request_queue: RequestQueue, callback_fn, thread_name="RequestHandler"):
        self.request_queue = request_queue
        self.callback_fn = callback_fn
        self.thread = None
        self.stop_event = threading.Event()
        self._sleep_interval = 0.3
        self.thread_name = thread_name

    def start(self):
        self.thread = threading.Thread(
            target=self._handle_requests_loop, daemon=True, name=self.thread_name
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def _handle_requests_loop(self):
        while not self.stop_event.is_set():
            requests = self.request_queue.get_all()
            for request in requests:
                self.callback_fn(request)
            time.sleep(self._sleep_interval)


def set_exception(future: concurrent.futures.Future, exception: Exception):
    # concurrent.futures.Future.set_exception is thread-safe.
    if not future.done():
        future.set_exception(exception)
    else:
        logger.warning(f"Future already done when setting exception: {exception}")


def set_result(future: concurrent.futures.Future, result):
    # concurrent.futures.Future.set_result is thread-safe.
    if not future.done():
        future.set_result(result)
    else:
        logger.warning(f"Future already done when setting result: {result}")
