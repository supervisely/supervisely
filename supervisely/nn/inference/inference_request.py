from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Union

from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class InferenceRequest:
    class Stage:
        CREATED = "created"
        PREPARING = "preparing"
        PREPROCESS = "preprocess"
        INFERENCE = "inference"
        POSTPROCESS = "postprocess"
        FINISHED = "finished"
        CANCELLED = "cancelled"
        ERROR = "error"

    def __init__(self, uuid_: str = None, ttl: Union[int, None] = 60 * 60):
        if uuid_ is None:
            uuid_ = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=f"{time.time()}").hex
        self._uuid = uuid_
        self._ttl = ttl
        self._lock = threading.Lock()
        self._stage = InferenceRequest.Stage.CREATED
        self._pending_results = []
        self._final_result = None
        self._exception = None
        self.stopped = False
        self.progress = Progress(
            message="Preparing model for inference...",
            total_cnt=1,
            need_info_log=True,
            log_extra={"inference_request_uuid": self._uuid},
        )
        self._created_at = time.monotonic()
        self._updated_at = self._created_at

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
    def final_result(self, result: Dict):
        with self._lock:
            self._final_result = result
            self.__updated()

    def add_results(self, results: List[Dict]):
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

    def __update_stage(self, refresh_progress: bool = True):
        progress_msg = self.progress.message
        if self._stage in (InferenceRequest.Stage.CREATED, InferenceRequest.Stage.PREPARING):
            progress_msg = "Preparing model for inference..."
        elif self._stage == InferenceRequest.Stage.PREPROCESS:
            progress_msg = "Preprocessing data..."
        elif self._stage == InferenceRequest.Stage.INFERENCE:
            progress_msg = "Running inference..."
        elif self._stage == InferenceRequest.Stage.POSTPROCESS:
            progress_msg = "Postprocessing data..."
        elif self._stage == InferenceRequest.Stage.FINISHED:
            progress_msg = "Inference finished"
            refresh_progress = False
        elif self._stage == InferenceRequest.Stage.CANCELLED:
            progress_msg = "Inference cancelled"
            refresh_progress = False
        elif self._stage == InferenceRequest.Stage.ERROR:
            progress_msg = "Error: " + str(self._exception)
            refresh_progress = False
        else:
            progress_msg = str(self._stage)
            refresh_progress = False

        self.progress.message = progress_msg
        if refresh_progress:
            self.progress.current = 0
        self.progress.report_progress(update_task_progress=False)

    @stage.setter
    def stage(self, stage: Union[InferenceRequest.Stage, str]):
        self._stage = stage
        self.__update_stage()
        self.__updated()

    @property
    def exception(self):
        return self._exception

    @exception.setter
    def exception(self, exception: Exception):
        self._exception = exception
        self.stage = InferenceRequest.Stage.ERROR
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
                self.stage = InferenceRequest.Stage.CANCELLED
            else:
                self.stage = InferenceRequest.Stage.FINISHED
