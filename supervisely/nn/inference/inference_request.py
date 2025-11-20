from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Dict, List, Tuple, Union

from supervisely._utils import rand_str
from supervisely.nn.utils import get_gpu_usage, get_ram_usage
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


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
        self.context = {}
        self._lock = threading.Lock()
        self._stage = InferenceRequest.Stage.PREPARING
        self._pending_results = []
        self._final_result = None
        self._exception = None
        self._stopped = threading.Event()
        self._progress_log_interval = 5.0
        self._last_progress_report_time = 0
        self.progress = Progress(
            message=self._stage,
            total_cnt=1,
            need_info_log=True,
            update_task_progress=False,
            log_extra={"inference_request_uuid": self._uuid},
        )
        self._created_at = time.monotonic()
        self._updated_at = self._created_at
        self._finished = False

        self.tracker = None

        self.global_progress = None
        self.global_progress_total = 1
        self.global_progress_current = 0

    def _updated(self):
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
            self._updated()

    def add_results(self, results: List[Dict]):
        with self._lock:
            self._pending_results.extend(results)
            self._updated()

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
            self._updated()
            return results

    def pending_num(self):
        return len(self._pending_results)

    def set_stage(self, stage: str, current: int = None, total: int = None, is_size: bool = False):
        with self._lock:
            self._stage = stage
            self.progress.message = self._stage
            if current is not None:
                self.progress.current = current
            if total is not None:
                logger.debug("setting total = %s", total)
                self.progress.total = total
            if is_size:
                self.progress.is_size = True
            self.progress._refresh_labels()
            self.progress.report_progress()
            if self._stage == InferenceRequest.Stage.INFERENCE:
                self.global_progress_total = total
                self.global_progress_current = current
                self.manager.global_progress.inference_started(
                    current=self.global_progress_current,
                    total=self.global_progress_total,
                )
            self._updated()

    def set_progress_log_interval(self, interval: float):
        self._progress_log_interval = interval

    def done(self, n=1):
        with self._lock:
            if (
                self._progress_log_interval is None
                or time.monotonic() - self._last_progress_report_time > self._progress_log_interval
            ) or (self.progress.current + n >= self.progress.total):
                self.progress.iters_done_report(n)
                self._last_progress_report_time = time.monotonic()
            else:
                self.progress.iters_done(n)
            if self._stage == InferenceRequest.Stage.INFERENCE:
                self.global_progress_current += n
                if self.manager is not None:
                    self.manager.done(n)
            self._updated()

    @property
    def exception(self):
        return self._exception

    @exception.setter
    def exception(self, exc: Exception):
        self._exception = exc
        self.set_stage(InferenceRequest.Stage.ERROR)
        self._updated()

    def is_inferring(self):
        return self.stage == InferenceRequest.Stage.INFERENCE

    def stop(self):
        self._stopped.set()
        self._updated()

    def is_stopped(self):
        return self._stopped.is_set()

    def is_finished(self):
        return self._finished

    def is_expired(self):
        if self._ttl is None:
            return False
        return time.monotonic() - self._updated_at > self._ttl

    def progress_json(self):
        return {
            "message": self.progress.message,
            "status": self.progress.message,
            "current": self.progress.current,
            "total": self.progress.total,
            "is_size": self.progress.is_size,
        }

    def exception_json(self):
        if self._exception is None:
            return None
        return {
            "type": type(self._exception).__name__,
            "message": str(self._exception),
            "traceback": str(traceback.format_exc()),
        }

    def to_json(self):
        return {
            "uuid": self._uuid,
            "stage": str(self._stage),
            "progress": self.progress_json(),
            "pending_results": self.pending_num(),
            "final_result": self._final_result is not None,
            "exception": self.exception_json(),
            "is_inferring": self.is_inferring(),
            "stopped": self.is_stopped(),
            "finished": self._finished,
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
        self._finished = True
        self._updated()

    def get_usage(self):
        ram_allocated, ram_total = get_ram_usage()
        gpu_allocated, gpu_total = get_gpu_usage()
        return {
            "gpu_memory": {
                "allocated": gpu_allocated,
                "total": gpu_total,
            },
            "ram_memory": {
                "allocated": ram_allocated,
                "total": ram_total,
            },
        }

    def status(self):
        status_data = self.to_json()
        for key in ["pending_results", "final_result", "created_at", "updated_at"]:
            status_data.pop(key, None)
        status_data.update(self.get_usage())
        return status_data


class GlobalProgress:
    def __init__(self):
        self.progress = Progress(message="Ready", total_cnt=1)
        self._lock = threading.Lock()

    def set_message(self, message: str):
        with self._lock:
            if self.progress.message != message:
                self.progress.message = message
                self.progress.report_progress()

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

    def inference_started(self, current: int, total: int):
        with self._lock:
            if self.progress.message == "Ready":
                self.progress.total = total
                self.progress.current = current
            else:
                self.progress.total += total
                self.progress.current += current
        self.set_message("Inferring model...")

    def inference_finished(self, current: int, total: int):
        with self._lock:
            if self.progress.message == "Ready":
                return
            self.progress.current = self.progress.current - current
            self.progress.total = self.progress.total - total
        if self.progress.current >= self.progress.total:
            self.set_ready()

    def to_json(self):
        return {
            "message": self.progress.message,
            "current": self.progress.current,
            "total": self.progress.total,
        }


class InferenceRequestsManager:

    def __init__(self, executor: ThreadPoolExecutor = None):
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
        self._executor = executor
        self._inference_requests: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self._monitor_thread.start()
        self.global_progress = GlobalProgress()

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
            self._stop_event.set()
            self._monitor_thread.join(timeout=5)
        finally:
            logger.debug("InferenceRequestsManager was deleted")

    def add(self, inference_request: InferenceRequest):
        with self._lock:
            self._inference_requests[inference_request.uuid] = inference_request

    def remove(self, inference_request_uuid: str):
        with self._lock:
            inference_request = self._inference_requests.get(inference_request_uuid)
            if inference_request is not None:
                inference_request.stop()
                del self._inference_requests[inference_request_uuid]

    def remove_after(self, inference_request_uuid, wait_time=0):
        with self._lock:
            inference_request = self._inference_requests.get(inference_request_uuid)
            if inference_request is not None:
                inference_request.stop()
                inference_request._ttl = wait_time
                inference_request._updated()

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
        while self._stop_event.is_set() is False:
            for inference_request_uuid in list(self._inference_requests.keys()):
                inference_request = self._inference_requests.get(inference_request_uuid)
                if inference_request is None:
                    continue
                if inference_request.is_expired():
                    self.remove(inference_request_uuid)
                    logger.debug(f"Expired inference request {inference_request_uuid} was deleted")
            time.sleep(30)

    def done(self, n=1):
        with self._lock:
            self.global_progress.done(n)

    def _on_inference_start(self, inference_request: InferenceRequest):
        if inference_request.uuid not in self._inference_requests:
            self.add(inference_request)

    def _on_inference_end(self, future, inference_request_uuid: str):
        logger.debug("callback: on_inference_end()")
        inference_request = self._inference_requests.get(inference_request_uuid)
        if inference_request is not None:
            inference_request.on_inference_end()

            self.global_progress.inference_finished(
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
            kwargs["inference_request"] = inference_request
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

    def run(self, func, *args, **kwargs):
        inference_request, future = self.schedule_task(func, *args, **kwargs)
        future.result()
        return inference_request.pop_pending_results()
