import functools
import inspect
import json
import traceback
from threading import Lock
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from fastapi import BackgroundTasks, Form, Request, Response, UploadFile, status

from supervisely.api.api import Api
from supervisely.api.entity_annotation.figure_api import FigureApi, FigureInfo
from supervisely.api.module_api import ApiField
from supervisely.io import env
from supervisely.nn.inference.inference import (
    Inference,
    _convert_sly_progress_to_dict,
    _get_log_extra_for_inference_request,
)
from supervisely.sly_logger import logger


class ValidationError(Exception):
    pass


def validate_key(data: Dict, key: str, type_: type):
    if key not in data:
        raise ValidationError(f"Key {key} not found in inference request.")
    if not isinstance(data[key], type_):
        raise ValidationError(f"Key {key} is not of type {type_}.")


def handle_validation(func):
    def _find_response(args, kwargs):
        for arg in args:
            if isinstance(arg, Response):
                return arg
        for value in kwargs.values():
            if isinstance(value, Response):
                return value
        return None

    def _handle_exception(e, response):
        if response is not None:
            logger.error(f"ValidationError: {e}", exc_info=True)
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"error": str(e), "success": False}
        raise e

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            response = _find_response(args, kwargs)
            try:
                return await func(*args, **kwargs)
            except ValidationError as e:
                return _handle_exception(e, response)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = _find_response(args, kwargs)
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            return _handle_exception(e, response)

    return wrapper


class BaseTracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        Inference.__init__(
            self,
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")
            logger.warning("Failed to load model on CUDA device.")

        logger.debug(
            "Smart cache params",
            extra={"ttl": env.smart_cache_ttl(), "maxsize": env.smart_cache_size()},
        )

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def _on_inference_start(self, inference_request_uuid: str):
        super()._on_inference_start(inference_request_uuid)
        self._inference_requests[inference_request_uuid]["lock"] = Lock()

    def _notify_error_default(
        self, api: Api, track_id: str, exception: Exception, with_traceback: bool = False
    ):
        error_name = type(exception).__name__
        message = str(exception)
        if with_traceback:
            message = f"{message}\n{traceback.format_exc()}"
        api.video.notify_tracking_error(track_id, error_name, message)

    def _notify_error_direct(
        self,
        api: Api,
        session_id: str,
        video_id,
        track_id: str,
        exception: Exception,
        with_traceback: bool = False,
    ):
        error_name = type(exception).__name__
        message = str(exception)
        if with_traceback:
            message = f"{message}\n{traceback.format_exc()}"
        api.vid_ann_tool.set_direct_tracking_error(
            session_id=session_id,
            video_id=video_id,
            track_id=track_id,
            message=f"{error_name}: {message}",
        )

    def send_error_data(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = None
            try:
                value = func(*args, **kwargs)
            except Exception as exc:
                try:
                    logger.error(f"An error occured: {exc}", exc_info=True)
                    request: Request = args[0]
                    context = request.state.context
                    api: Api = request.state.api
                    track_id = context["trackId"]
                    if ApiField.USE_DIRECT_PROGRESS_MESSAGES in context:
                        session_id = context.get("sessionId", context.get("session_id", None))
                        video_id = context.get("videoId", context.get("video_id", None))
                        self._notify_error_direct(
                            api=api,
                            session_id=session_id,
                            video_id=video_id,
                            track_id=track_id,
                            exception=exc,
                            with_traceback=False,
                        )
                    else:
                        self._notify_error_default(
                            api=api, track_id=track_id, exception=exc, with_traceback=False
                        )
                except Exception:
                    logger.error("An error occured while sending error data", exc_info=True)
            return value

        return wrapper

    def _figure_info_to_json(self, figure: FigureInfo):
        return FigureApi.convert_info_to_json(figure)

    def _pop_tracking_results(self, inference_request_uuid: str, frame_range: Tuple = None):
        inference_request = self._inference_requests[inference_request_uuid]
        logger.debug(
            "Pop tracking results",
            extra={
                "inference_request_uuid": inference_request_uuid,
                "pending_results_len": len(inference_request["pending_results"]),
                "pending_results[:3]": inference_request["pending_results"][:3],
                "frame_range": frame_range,
            },
        )
        logger.debug("Acquiring lock for pop")
        # with inference_request["lock"]:
        inference_request_copy = inference_request.copy()

        if frame_range is not None:

            def _in_range(figure):
                return figure.frame_index >= frame_range[0] and figure.frame_index <= frame_range[1]

            inference_request_copy["pending_results"] = list(
                filter(_in_range, inference_request_copy["pending_results"])
            )
            inference_request["pending_results"] = list(
                filter(lambda x: not _in_range(x), inference_request["pending_results"])
            )
        else:
            inference_request["pending_results"] = []
        # lock
        logger.debug("Released lock for pop")
        inference_request_copy.pop("lock")
        inference_request_copy["progress"] = _convert_sly_progress_to_dict(
            inference_request_copy["progress"]
        )

        inference_request_copy["pending_results"] = [
            self._figure_info_to_json(figure)
            for figure in inference_request_copy["pending_results"]
        ]

        return inference_request_copy

    def _clear_tracking_results(self, inference_request_uuid):
        del self._inference_requests[inference_request_uuid]
        logger.debug("Removed an inference request:", extra={"uuid": inference_request_uuid})

    def _stop_tracking(self, inference_request_uuid: str):
        inference_request = self._inference_requests[inference_request_uuid]
        inference_request["cancel_inference"] = True
        logger.debug("Stopped tracking:", extra={"uuid": inference_request_uuid})

    def track(self, api: Api, state: Dict, context: Dict):
        raise NotImplementedError("Method `track` must be implemented.")

    def track_api(self, api: Api, state: Dict, context: Dict):
        raise NotImplementedError("Method `_track_api` must be implemented.")

    def track_api_files(
        self,
        files: List[BinaryIO],
        settings: Dict,
    ):
        raise NotImplementedError("Method `track_api_files` must be implemented.")

    def track_async(self, api: Api, state: Dict, context: Dict):
        raise NotImplementedError("Method `track_async` must be implemented.")

    def stop_tracking(self, state: Dict, context: Dict):
        validate_key(context, "inference_request_uuid", str)
        inference_request_uuid = context["inference_request_uuid"]
        self._stop_tracking(inference_request_uuid)
        return {"message": "Inference will be stopped.", "success": True}

    def pop_tracking_results(self, state: Dict, context: Dict):
        validate_key(context, "inference_request_uuid", str)
        inference_request_uuid = context["inference_request_uuid"]
        frame_range = context.get("frameRange", context.get("frame_range", None))
        tracking_results = self._pop_tracking_results(inference_request_uuid, frame_range)
        log_extra = _get_log_extra_for_inference_request(inference_request_uuid, tracking_results)
        logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
        return tracking_results

    def clear_tracking_results(self, state: Dict, context: Dict):
        self._clear_tracking_results(context)
        return {"message": "Inference results cleared.", "success": True}

    def _register_endpoints(self):
        server = self._app.get_server()

        @server.post("/track")
        @handle_validation
        def track_handler(request: Request):
            api = request.state.api
            state = request.state.state
            context = request.state.context
            logger.info("Received track request.", extra={"context": context, "state": state})
            self.schedule_task(self.track, api, state, context)
            return {"message": "Track task started."}

        @server.post("/track-api")
        @handle_validation
        async def track_api_handler(request: Request):
            api = request.state.api
            state = request.state.state
            context = request.state.context
            logger.info("Received track-api request.", extra={"context": context, "state": state})
            result = self.track_api(api, state, context)
            logger.info("Track-api request processed.")
            return result

        @server.post("/track-api-files")
        @handle_validation
        def track_api_files(
            files: List[UploadFile],
            settings: str = Form("{}"),
        ):
            files = [file.file for file in files]
            settings = json.loads(settings)
            return self.track_api_files(files, settings)

        @server.post("/track_async")
        @handle_validation
        def track_async_handler(request: Request):
            api = request.state.api
            state = request.state.state
            context = request.state.context
            logger.info("Received track_async request.", extra={"context": context, "state": state})
            return self.track_async(api, state, context)

        @server.post("/stop_tracking")
        @handle_validation
        def stop_tracking_handler(response: Response, request: Request):
            state = request.state.state
            context = request.state.context
            logger.info(
                "Received stop_tracking request.", extra={"context": context, "state": state}
            )
            return self.stop_tracking(state, context)

        @server.post("/pop_tracking_results")
        @handle_validation
        def pop_tracking_results_handler(request: Request, response: Response):
            state = request.state.state
            context = request.state.context
            logger.info(
                "Received pop_tracking_results request.", extra={"context": context, "state": state}
            )
            return self.pop_tracking_results(state, context)

        @server.post("/clear_tracking_results")
        @handle_validation
        def clear_tracking_results_handler(request: Request, response: Response):
            context = request.state.context
            state = request.state.state
            logger.info(
                "Received clear_tracking_results request.",
                extra={"context": context, "state": state},
            )
            return self.clear_tracking_results(state, context)

    def serve(self):
        super().serve()
        self._register_endpoints()
