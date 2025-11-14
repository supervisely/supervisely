from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from logging import Logger
from typing import Callable, List

from supervisely.sly_logger import logger as sly_logger


class Uploader:
    def __init__(
        self,
        upload_f: Callable,
        notify_f: Callable = None,
        exception_handler: Callable = None,
        logger: Logger = None,
    ):
        self._upload_f = upload_f
        self._notify_f = notify_f
        self._exception_handler = exception_handler
        if self._exception_handler is None:
            self._exception_handler = self._default_exception_handler
        self._logger = logger
        self.exception = None
        self._lock = threading.Lock()
        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._exception_event = threading.Event()
        self._upload_thread = threading.Thread(
            target=self._upload_loop,
            daemon=True,
        )
        self.raise_from_notify = False
        self._notify_thread = None
        self._notify_q = queue.Queue()
        if self._notify_f is not None:
            self._notify_thread = threading.Thread(target=self._notify_loop, daemon=True)
            self._notify_thread.start()
        self._upload_thread.start()

    def _notify_loop(self):
        try:
            while (
                not self._stop_event.is_set() or not self._notify_q.empty() or not self._q.empty()
            ):
                if self._exception_event.is_set():
                    return
                items = []
                try:
                    timeout = 0.1 if self._stop_event.is_set() else 1.0
                    item = self._notify_q.get(timeout=timeout)
                    items.append(item)
                    while True:
                        try:
                            items.append(self._notify_q.get_nowait())
                        except queue.Empty:
                            break
                    if items:
                        self._notify_f(items)

                    for _ in range(len(items)):
                        self._notify_q.task_done()
                except queue.Empty:
                    continue
        except StopIteration:
            self.stop()
            return
        except Exception as e:
            try:
                raise RuntimeError("Error in notify loop") from e
            except RuntimeError as e_:
                e = e_
            if self._logger is not None:
                self._logger.error("Error in notify loop: %s", str(e), exc_info=True)
            if self.raise_from_notify and not self._exception_event.is_set():
                self.set_exception(e)
            return

    def _upload_loop(self):
        try:
            while not self._stop_event.is_set() or not self._q.empty():
                if self._exception_event.is_set():
                    return
                items = []
                try:
                    timeout = 0.1 if self._stop_event.is_set() else 1.0
                    item = self._q.get(timeout=timeout)
                    items.append(item)
                    while True:
                        try:
                            items.append(self._q.get_nowait())
                        except queue.Empty:
                            break
                    if items:
                        self._upload_f(items)
                        self.notify(items)

                    for _ in range(len(items)):
                        self._q.task_done()
                except queue.Empty:
                    continue
        except StopIteration:
            self.stop()
            return
        except Exception as e:
            if self._logger is not None:
                self._logger.error("Error in upload loop: %s", str(e), exc_info=True)
            if not self._exception_event.is_set():
                self.set_exception(e)
            return

    def put(self, items):
        for item in items:
            self._q.put(item)

    def notify(self, items):
        if self._notify_thread is not None:
            for item in items:
                self._notify_q.put(item)

    def stop(self):
        self._stop_event.set()

    def join(self, timeout=None):
        self._upload_thread.join(timeout=timeout)
        if self._notify_thread is not None:
            self._notify_thread.join(timeout=timeout)

    def has_exception(self):
        return self._exception_event.is_set()

    def set_exception(self, exception: Exception):
        with self._lock:
            self._exception_event.set()
            self.exception = exception

    def __enter__(self):
        return self

    def _default_exception_handler(
        self,
        exception: Exception,
    ):
        return False  # propagate

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            self.join(timeout=30)
            if self._upload_thread.is_alive():
                raise TimeoutError("Uploader thread didn't finish in time")
        except TimeoutError:
            _logger = sly_logger
            if self._logger is not None:
                _logger = self._logger
            _logger.warning("Uploader thread didn't finish in time")
        if exc_type is not None:
            exc = exc_val.with_traceback(exc_tb)
            return self._exception_handler(exc)
        if self.has_exception():
            exc = self.exception
            try:
                raise RuntimeError(f"Error in uploader loop: {str(exc)}") from exc
            except Exception as exc:
                return self._exception_handler(exc)
        return False


class Downloader:

    def __init__(
        self,
        download_f: Callable,
        max_workers: int = 8,
        buffer_size: int = 100,
        exception_handler: Callable = None,
        logger: Logger = None,
    ):
        self._download_f = download_f
        self._max_workers = max_workers
        self._logger = logger
        self._exception_handler = exception_handler
        if self._exception_handler is None:
            self._exception_handler = self._default_exception_handler
        self._input_q = queue.Queue()
        self._buffer_q = queue.Queue(buffer_size)
        self._output_q = queue.Queue()
        self._executor: ThreadPoolExecutor = None
        self._download_futures: List[Future] = None
        self._stop_event = threading.Event()

    def _download_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._buffer_q.get(timeout=0.2)
                try:
                    output = self._download_f(item)
                    self._output_q.put(output)
                finally:
                    self._buffer_q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger = self._logger or sly_logger
                logger.debug("Error in downloader thread", exc_info=True)

    def start(self):
        if self.is_alive():
            raise RuntimeError("Downloader already started")
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._download_futures = []
        for _ in range(self._max_workers):
            self._download_futures.append(self._executor.submit(self._download_loop))

    def put(self, item):
        self._input_q.put(item)

    def get(self, wait=True, timeout: float = None):
        return self._output_q.get(block=wait, timeout=timeout)

    def _move_input_to_buffer(self):
        try:
            item = self._input_q.get_nowait()
        except queue.Empty:
            return
        for _ in range(10):
            try:
                self._buffer_q.put_nowait(item)
                return
            except queue.Full:
                pass
            try:
                self._buffer_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._buffer_q.put_nowait(item)
            return
        except:
            raise RuntimeError("Unable to move item from input to buffer")

    def next(self, n: int = 1, raise_on_error=False):
        for _ in range(n):
            try:
                self._move_input_to_buffer()
            except Exception:
                if raise_on_error:
                    raise
                logger = sly_logger
                if self._logger is not None:
                    logger = self._logger
                logger.debug("Error moving buffer", exc_info=True)
                continue

    def is_alive(self):
        return self._executor is not None and any(not f.done() for f in self._download_futures)

    def stop(self):
        self._stop_event.set()
        for future in self._download_futures:
            future.cancel()
        self._executor.shutdown(wait=False)

    def join(self, timeout=None):
        _, not_done = wait(self._download_futures, timeout=timeout)
        if not_done:
            raise TimeoutError("Timeout waiting for downloads to complete")

    def __enter__(self):
        self.start()
        return self

    def _default_exception_handler(
        self,
        exception: Exception,
    ):
        return False  # propagate

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            self.join(timeout=30)
            if self.is_alive():
                raise TimeoutError("Downloader threads didn't finish in time")
        except TimeoutError:
            _logger = sly_logger
            if self._logger is not None:
                _logger = self._logger
            _logger.warning("Downloader threads didn't finish in time")
        if exc_type is not None:
            exc = exc_val.with_traceback(exc_tb)
            return self._exception_handler(exc)
        return False
