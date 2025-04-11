from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple, Union

from supervisely._utils import rand_str
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


def generate_uuid(self) -> str:
    """
    Generates a unique UUID for the inference request.
    """
    return uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}-{rand_str(10)}").hex


class InferenceRequest:
    class Stage:
        PREPARING = "Preparing model for inference..."
        INFERENCE = "Running inference..."
        FINISHED = "Finished"
        CANCELLED = "Cancelled"
        ERROR = "Error"

    def __init__(
        self,
        uuid_: str = None,
        ttl: Union[int, None] = 60 * 60,
        manager: InferenceRequestsManager = None,
    ):
        if uuid_ is None:
            uuid_ = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
        self._uuid = uuid_
        self._ttl = ttl
        self.manager = manager
        self.save_results = True
        self._lock = threading.Lock()
        self._stage = InferenceRequest.Stage.PREPARING
        self._pending_results = []
        self._final_result = None
        self._exception = None
        self.stopped = False
        self.progress = Progress(
            message=self._stage,
            total_cnt=1,
            need_info_log=True,
            update_task_progress=False,
            log_extra={"inference_request_uuid": self._uuid},
        )
        self._created_at = time.monotonic()
        self._updated_at = self._created_at

        self.global_progress = None
        self.global_progress_total = 1
        self.global_progress_current = 0

    def __updated(self):
        self._updated_at = time.monotonic()

    @property
    def uuid(self):
        return self._uuid

    @property
    def created_at(self):
        return self._created_at

    @property
    def updated_at(self):
        return self._updated_at

    @property
    def stage(self):
        return self._stage

    @property
    def final_result(self):
        with self._lock:
            return self._final_result

    @final_result.setter
    def final_result(self, result: Any):
        with self._lock:
            self._final_result = result
            self.__updated()

    def add_results(self, results: List[Dict]):
        if self.save_results:
            with self._lock:
                self._pending_results.extend(results)
                self.__updated()

    def pop_pending_results(self, n: int = None):
        with self._lock:
            if len(self._pending_results) == 0:
                return []
            if n is None:
                n = len(self._pending_results)
            if n > len(self._pending_results):
                n = len(self._pending_results)
            results = self._pending_results[:n]
            self._pending_results = self._pending_results[n:]
            self.__updated()
            return results

    def pending_num(self):
        return len(self._pending_results)

    def set_stage(self, stage: str, current: int = None, total: int = None):
        with self._lock:
            self._stage = stage
            self.progress.message = self._stage
            if current is not None:
                self.progress.current = current
            if total is not None:
                self.progress.total = total
            self.progress.report_progress()
            if self._stage == InferenceRequest.Stage.INFERENCE:
                self.global_progress_total = total
                self.global_progress_current = current

    def done(self, n=1):
        with self._lock:
            self.progress.iters_done_report(n)
            if self._stage == InferenceRequest.Stage.INFERENCE:
                self.global_progress_current += n
                if self.manager is not None:
                    self.manager.done(n)

    @property
    def exception(self):
        return self._exception

    @exception.setter
    def exception(self, exc: Exception):
        self._exception = exc
        with self._lock:
            self.set_stage(InferenceRequest.Stage.ERROR)
        self.__updated()

    def is_inferring(self):
        return self.stage == InferenceRequest.Stage.INFERENCE

    def stop(self):
        self.stopped = True
        self.__updated()

    def is_stopped(self):
        return self.stopped

    def is_expired(self):
        if self._ttl is None:
            return False
        return time.monotonic() - self._updated_at > self._ttl

    def progress_json(self):
        return {
            # "message": self.progress.message,
            "current": self.progress.current,
            "total": self.progress.total,
        }

    def exception_json(self):
        if self._exception is None:
            return None
        return {
            "type": str(type(self._exception)),
            "message": str(self._exception),
            "traceback": str(self._exception.__traceback__),
        }

    def to_json(self):
        return {
            "uuid": self._uuid,
            "stage": str(self._stage),
            "progress": self.progress_json(),
            "pending_results": self.pending_num(),
            "final_result": self.final_result,
            "exception": self.exception_json(),
            "stopped": self.stopped,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }

    def on_inference_end(self):
        if self._stage not in (
            InferenceRequest.Stage.FINISHED,
            InferenceRequest.Stage.CANCELLED,
            InferenceRequest.Stage.ERROR,
        ):
            if self.is_stopped():
                self.set_stage(InferenceRequest.Stage.CANCELLED)
            else:
                self.set_stage(InferenceRequest.Stage.FINISHED)


class GlobalProgress:
    def __init__(self):
        self.progress = Progress(message="Ready", total_cnt=1)
        self._lock = threading.Lock()

    def set_message(self, message: str):
        with self._lock:
            if self.progress.message != message:
                self.progress.message = message
                self.progress.report_progress()

    def increase_total(self, n=1):
        with self._lock:
            if (
                self.progress.message != "Inference in progress..."
                and self.progress.current == 0
                and self.progress.total == 1
            ):
                self.progress.total = n
            else:
                self.progress.total += n
        self.set_message("Inference in progress...")

    def set_ready(self):
        with self._lock:
            self.progress.message = "Ready"
            self.progress.current = 0
            self.progress.total = 1
            self.progress.report_progress()

    def done(self, n=1):
        with self._lock:
            self.progress.iters_done_report(n)
        if self.progress.current >= self.progress.total:
            self.set_ready()

    def inference_finished(self, current: int, total: int):
        with self._lock:
            if self.progress.message == "Ready":
                return
            self.progress.current = max(0, self.progress.current - current)
            self.progress.total = max(1, self.progress.total - total)


class InferenceRequestsManager:

    def __init__(self, executor: ThreadPoolExecutor = None):
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
        self._executor = executor
        self._inference_requests: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()
        self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self._monitor_thread.start()
        self._global_progress = GlobalProgress()

    def add(self, inference_request: InferenceRequest):
        with self._lock:
            self._inference_requests[inference_request.uuid] = inference_request

    def remove(self, uuid_: str):
        with self._lock:
            if uuid_ in self._inference_requests:
                del self._inference_requests[uuid_]

    def get(self, inference_request_uuid: str):
        if inference_request_uuid is None:
            return None
        try:
            return self._inference_requests[inference_request_uuid]
        except Exception as ex:
            raise RuntimeError(
                f"inference_request_uuid {inference_request_uuid} was given, "
                f"but there is no such uuid in 'self._inference_requests' ({len(self._inference_requests)} items)"
            ) from ex

    def create(self, inference_request_uuid: str = None) -> InferenceRequest:
        inference_request = InferenceRequest(uuid_=inference_request_uuid, manager=self)
        self.add(inference_request)
        return inference_request

    def monitor(self):
        while True:
            for inference_request in self._inference_requests.values():
                if inference_request.is_expired():
                    inference_request_uuid = inference_request.uuid
                    self.remove(inference_request_uuid)
                    logger.debug(f"Expired inference request {inference_request_uuid} was deleted")
            time.sleep(30)

    def done(self, n=1):
        with self._lock:
            self._global_progress.done(n)

    def _on_inference_start(self, inference_request: InferenceRequest):
        if inference_request.uuid not in self._inference_requests:
            self.add(inference_request)

    def _on_inference_end(self, future, inference_request_uuid: str):
        logger.debug("callback: on_inference_end()")
        inference_request = self._inference_requests.get(inference_request_uuid)
        if inference_request is not None:
            inference_request.on_inference_end()

            self._global_progress.inference_finished(
                current=inference_request.global_progress_current,
                total=inference_request.global_progress_total,
            )

    def _handle_error_in_async(self, inference_request_uuid: str, func, args, kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            inference_request = self._inference_requests.get(inference_request_uuid, None)
            if inference_request is not None:
                inference_request.exception = e
            logger.error(f"Error in {func.__name__} function: {e}", exc_info=True)

    def schedule_task(self, func, *args, **kwargs) -> Tuple[InferenceRequest, Future]:
        inference_request = kwargs.get("inference_request", None)
        if inference_request is None:
            inference_request = self.create()
        self._on_inference_start(inference_request)
        future = self._executor.submit(
            self._handle_error_in_async,
            inference_request.uuid,
            func,
            args,
            kwargs,
        )
        end_callback = partial(
            self._on_inference_end, inference_request_uuid=inference_request.uuid
        )
        future.add_done_callback(end_callback)
        logger.debug("Scheduled task.", extra={"inference_request_uuid": inference_request.uuid})
        return inference_request, future
