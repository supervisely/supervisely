from __future__ import annotations

import threading
from logging import Logger
from queue import Empty, Queue
from types import TracebackType
from typing import Callable, Optional, Type

from supervisely.sly_logger import logger


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
        self._q = Queue()
        self._stop_event = threading.Event()
        self._exception_event = threading.Event()
        self._upload_thread = threading.Thread(
            target=self._upload_loop,
            daemon=True,
        )
        self.raise_from_notify = False
        self._notify_thread = None
        self._notify_q = Queue()
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
                        except Empty:
                            break
                    if items:
                        self._notify_f(items)

                    for _ in range(len(items)):
                        self._notify_q.task_done()
                except Empty:
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
                        except Empty:
                            break
                    if items:
                        self._upload_f(items)
                        self.notify(items)

                    for _ in range(len(items)):
                        self._q.task_done()
                except Empty:
                    continue
        except StopIteration:
            self.stop()
            return
        except Exception as e:
            try:
                raise RuntimeError("Error in upload loop") from e
            except RuntimeError as e_:
                e = e_
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
        raise exception

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        try:
            self.join(timeout=5)
        except TimeoutError:
            _logger = logger
            if self._logger is not None:
                _logger = self._logger
            _logger.warning("Uploader thread didn't finish in time")
        if exc_type is not None:
            exc = exc_val.with_traceback(exc_tb)
            return self._exception_handler(exc)
        return False
