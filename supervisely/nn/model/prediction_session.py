import json
import time
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Tuple, Union

import numpy as np
import requests
from requests import Timeout
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm.auto import tqdm

import supervisely.io.env as env
from supervisely._utils import get_valid_kwargs, logger
from supervisely.api.api import Api
from supervisely.imaging._video import ALLOWED_VIDEO_EXTENSIONS
from supervisely.imaging.image import SUPPORTED_IMG_EXTS, write_bytes
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_size,
    list_files,
    list_files_recursively,
)
from supervisely.io.network_exceptions import process_requests_exception
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta


def value_generator(value):
    while True:
        yield value


class PredictionSession:

    class Iterator:
        def __init__(self, total, session: "PredictionSession", tqdm: tqdm = None):
            self.total = total
            self.session = session
            self.results_queue = []
            self.tqdm = tqdm

        def __len__(self) -> int:
            return self.total

        def __iter__(self) -> Iterator:
            return self

        def __next__(self) -> Dict[str, Any]:
            if not self.results_queue:
                pending_results = self.session._wait_for_pending_results(tqdm=self.tqdm)
                self.results_queue += pending_results
            if not self.results_queue:
                raise StopIteration
            pred = self.results_queue.pop(0)
            return pred

    def __init__(
        self,
        url: str,
        input: Union[np.ndarray, str, PathLike, list] = None,
        image_id: Union[List[int], int] = None,
        video_id: Union[List[int], int] = None,
        dataset_id: Union[List[int], int] = None,
        project_id: Union[List[int], int] = None,
        api: "Api" = None,
        tracking: bool = None,
        tracking_config: dict = None,
        **kwargs: dict,
    ): 

        extra_input_args = ["image_ids", "video_ids", "dataset_ids", "project_ids"]
        assert (
            sum(
                [
                    x is not None
                    for x in [
                        input,
                        image_id,
                        video_id,
                        dataset_id,
                        project_id,
                        *[kwargs.get(extra_input, None) for extra_input in extra_input_args],
                    ]
                ]
            )
            == 1
        ), "Exactly one of input, image_ids, video_id, dataset_id, project_id or image_id must be provided."

        self._iterator = None
        self._base_url = url
        self.inference_request_uuid = None
        self.input = input
        self.api = api

        self.api_token = self._get_api_token()
        self._model_meta = None
        self.final_result = None

        if "stride" in kwargs:
            kwargs["step"] = kwargs["stride"]
        if "start_frame" in kwargs:
            kwargs["start_frame_index"] = kwargs["start_frame"]
        if "num_frames" in kwargs:
            kwargs["frames_count"] = kwargs["num_frames"]
        self.kwargs = kwargs
        if kwargs.get("show_progress", False) and "tqdm" not in kwargs:
            kwargs["tqdm"] = tqdm()
        self.tqdm = kwargs.pop("tqdm", None)

        self.inference_settings = {
            k: v for k, v in kwargs.items() if isinstance(v, (str, int, float))
        }

        if tracking is True:
            model_info = self._get_session_info()
            if not model_info.get("tracking_on_videos_support", False):
                raise ValueError("Tracking is not supported by this model")

            if tracking_config is None:
                self.tracker = "botsort"
                self.tracker_settings = {}
            else:
                cfg = dict(tracking_config)
                self.tracker = cfg.pop("tracker", "botsort")
                self.tracker_settings = cfg
        else:
            self.tracker = None
            self.tracker_settings = None

        if "classes" in kwargs:
            self.inference_settings["classes"] = kwargs["classes"]
        # TODO: remove "settings", it is the same as inference_settings
        if "settings" in kwargs:
            self.inference_settings.update(kwargs["settings"])
        if "inference_settings" in kwargs:
            self.inference_settings.update(kwargs["inference_settings"])

        # extra input args
        image_ids = self._set_var_from_kwargs("image_ids", kwargs, image_id)
        video_ids = self._set_var_from_kwargs("video_ids", kwargs, video_id)
        dataset_ids = self._set_var_from_kwargs("dataset_ids", kwargs, dataset_id)
        project_ids = self._set_var_from_kwargs("project_ids", kwargs, project_id)
        source = next(
            x
            for x in [
                input,
                image_id,
                video_id,
                dataset_id,
                project_id,
                image_ids,
                video_ids,
                dataset_ids,
                project_ids,
            ]
            if x is not None
        )
        self.kwargs["source"] = source
        self.prediction_kwargs_iterator = value_generator({})

        if not isinstance(input, list):
            input = [input]
        if isinstance(input[0], np.ndarray):
            # input is numpy array
            self._predict_images(input, **kwargs)
        elif isinstance(input[0], (str, PathLike)):
            if len(input) > 1:
                # if the input is a list of paths, assume they are images
                for x in input:
                    if not isinstance(x, (str, PathLike)):
                        raise ValueError("Input must be a list of strings or PathLike objects.")
                self._iterator = self._predict_images_bytes(input, **kwargs)
            else:
                if dir_exists(input[0]):
                    try:
                        project = Project(str(input[0]), mode=OpenMode.READ)
                    except Exception:
                        project = None
                    image_paths = []
                    if project is not None:
                        for dataset in project.datasets:
                            dataset: Dataset
                            for _, image_path, _ in dataset.items():
                                image_paths.append(image_path)
                    else:
                        # if the input is a directory, assume it contains images
                        recursive = kwargs.get("recursive", False)
                        if recursive:
                            image_paths = list_files_recursively(
                                input[0], valid_extensions=SUPPORTED_IMG_EXTS
                            )
                        else:
                            image_paths = list_files(input[0], valid_extensions=SUPPORTED_IMG_EXTS)
                    if len(image_paths) == 0:
                        raise ValueError("Directory is empty.")
                    self._iterator = self._predict_images(image_paths, **kwargs)
                elif file_exists(input[0]):
                    ext = get_file_ext(input[0])
                    if ext == "":
                        raise ValueError("File has no extension.")
                    if ext.lower() in SUPPORTED_IMG_EXTS:
                        self._iterator = self._predict_images(input, **kwargs)
                    elif ext.lower() in ALLOWED_VIDEO_EXTENSIONS:
                        kwargs = get_valid_kwargs(kwargs, self._predict_videos, exclude=["videos"])
                        self._iterator = self._predict_videos(input, tracker=self.tracker, tracker_settings=self.tracker_settings, **kwargs)
                    else:
                        raise ValueError(
                            f"Unsupported file extension: {ext}. Supported extensions are: {SUPPORTED_IMG_EXTS + ALLOWED_VIDEO_EXTENSIONS}"
                        )
                else:
                    raise ValueError(f"File or directory does not exist: {input[0]}")
        elif image_ids is not None:
            self._iterator = self._predict_images(image_ids, **kwargs)
        elif video_ids is not None:
            if len(video_ids) > 1:
                raise ValueError("Only one video id can be provided.")
            kwargs = get_valid_kwargs(kwargs, self._predict_videos, exclude=["videos"])
            self._iterator = self._predict_videos(video_ids, tracker=self.tracker, tracker_settings=self.tracker_settings, **kwargs)
        elif dataset_ids is not None:
            kwargs = get_valid_kwargs(
                kwargs,
                self._predict_datasets,
                exclude=["dataset_ids"],
            )
            self._iterator = self._predict_datasets(dataset_ids, **kwargs)
        elif project_ids is not None:
            if len(project_ids) > 1:
                raise ValueError("Only one project id can be provided.")
            kwargs = get_valid_kwargs(
                kwargs,
                self._predict_projects,
                exclude=["project_ids"],
            )
            self._iterator = self._predict_projects(project_ids, **kwargs)
        else:
            raise ValueError(
                "Unknown input type. Supported types are: numpy array, path to a file or directory, ImageInfo, VideoInfo, ProjectInfo, DatasetInfo."
            )

    def _set_var_from_kwargs(self, key, kwargs, default):
        value = kwargs.get(key, default)
        if value is None:
            return None
        if not isinstance(value, list):
            value = [value]
        return value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type is not None:
            return False

    def __next__(self):
        try:
            prediction_json = self._iterator.__next__()
            this_kwargs = next(self.prediction_kwargs_iterator)
            prediction = Prediction.from_json(
                prediction_json, **self.kwargs, **this_kwargs, model_meta=self.model_meta
            )
            return prediction
        except StopIteration:
            self._on_infernce_end()
            raise
        except Exception:
            self.stop()
            raise

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterator)

    def _get_api_token(self):
        if self.api is not None:
            return self.api.token
        return env.api_token(raise_not_found=False)

    def _get_json_body(self):
        body = {"state": {}, "context": {}}
        if self.inference_request_uuid is not None:
            body["state"]["inference_request_uuid"] = self.inference_request_uuid
        if self.inference_settings:
            body["state"]["settings"] = self.inference_settings
        if self.api_token is not None:
            body["api_token"] = self.api_token
        if "model_prediction_suffix" in self.kwargs:
            body["state"]["model_prediction_suffix"] = self.kwargs["model_prediction_suffix"]
        return body

    def _post(self, method, *args, retries=5, **kwargs) -> requests.Response:
        if kwargs.get("headers") is None:
            kwargs["headers"] = {}
        if self.api is not None:
            retries = min(self.api.retry_count, retries)
            if "x-api-key" not in kwargs["headers"]:
                kwargs["headers"]["x-api-key"] = self.api.token
        url = self._base_url.rstrip("/") + "/" + method.lstrip("/")
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60
        for retry_idx in range(retries):
            response = None
            try:
                logger.trace(f"POST {url}")
                response = requests.post(url, *args, **kwargs)
                if response.status_code != requests.codes.ok:  # pylint: disable=no-member
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(
                    logger,
                    exc,
                    method,
                    url,
                    verbose=True,
                    swallow_exc=True,
                    sleep_sec=5,
                    response=response,
                    retry_info={"retry_idx": retry_idx + 1, "retry_limit": retries},
                )
                if retry_idx + 1 == retries:
                    raise exc

    def _get_session_info(self) -> Dict[str, Any]:
        method = "get_session_info"
        r = self._post(method, json=self._get_json_body())
        return r.json()

    def _get_inference_progress(self):
        method = "get_inference_progress"
        r = self._post(method, json=self._get_json_body())
        return r.json()

    def _get_inference_status(self):
        method = "get_inference_status"
        r = self._post(method, json=self._get_json_body())
        return r.json()

    def _stop_async_inference(self):
        method = "stop_inference"
        r = self._post(
            method,
            json=self._get_json_body(),
        )
        logger.info("Inference will be stopped on the server")
        return r.json()

    def _clear_inference_request(self):
        method = "clear_inference_request"
        r = self._post(
            method,
            json=self._get_json_body(),
        )
        logger.info("Inference request will be cleared on the server")
        return r.json()

    def _get_final_result(self):
        method = "get_inference_result"
        r = self._post(
            method,
            json=self._get_json_body(),
        )
        return r.json()

    def _on_infernce_end(self):
        if self.inference_request_uuid is None:
            return
        try:
            self.final_result = self._get_final_result()
        except Exception as e:
            logger.debug("Failed to get final result:", exc_info=True)
        self._clear_inference_request()

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            self._model_meta = ProjectMeta.from_json(
                self._post("get_model_meta", json=self._get_json_body()).json()
            )
        return self._model_meta

    def stop(self):
        if self.inference_request_uuid is None:
            logger.debug("No active inference request to stop.")
            return
        self._stop_async_inference()
        self._on_infernce_end()

    def is_done(self):
        if self.inference_request_uuid is None:
            raise RuntimeError(
                "Inference is not started. Please start inference before checking the status."
            )
        return not self._get_inference_progress()["is_inferring"]

    def progress(self):
        if self.inference_request_uuid is None:
            raise RuntimeError(
                "Inference is not started. Please start inference before checking the status."
            )
        return self._get_inference_progress()["progress"]

    def status(self):
        if self.inference_request_uuid is None:
            raise RuntimeError(
                "Inference is not started. Please start inference before checking the status."
            )
        return self._get_inference_status()

    def _pop_pending_results(self) -> Dict[str, Any]:
        method = "pop_inference_results"
        json_body = self._get_json_body()
        return self._post(method, json=json_body).json()

    def _update_progress(
        self,
        tqdm: tqdm,
        message: str = None,
        current: int = None,
        total: int = None,
        is_size: bool = False,
    ):
        if tqdm is None:
            return
        refresh = False
        if message is not None and tqdm.desc != message:
            tqdm.set_description(message, refresh=False)
            refresh = True
        if current is not None and tqdm.n != current:
            tqdm.n = current
            refresh = True
        if total is not None and tqdm.total != total:
            tqdm.total = total
            refresh = True
        if is_size and tqdm.unit == "it":
            tqdm.unit = "iB"
            tqdm.unit_scale = True
            tqdm.unit_divisor = 1024
            refresh = True
        if not is_size and tqdm.unit == "iB":
            tqdm.unit = "it"
            tqdm.unit_scale = False
            tqdm.unit_divisor = 1
            refresh = True
        if refresh:
            tqdm.refresh()

    def _update_progress_from_response(self, tqdm: tqdm, response: Dict[str, Any]):
        if tqdm is None:
            return
        json_progress = response.get("progress", None)
        if json_progress is None or json_progress.get("message") is None:
            json_progress = response.get("preparing_progress", None)
        if json_progress is None:
            return
        message = json_progress.get("message", json_progress.get("status", None))
        current = json_progress.get("current", None)
        total = json_progress.get("total", None)
        is_size = json_progress.get("is_size", False)
        self._update_progress(tqdm, message, current, total, is_size)

    def _wait_for_inference_start(
        self, delay=1, timeout=None, tqdm: tqdm = None
    ) -> Tuple[dict, bool]:
        has_started = False
        timeout_exceeded = False
        t0 = time.time()
        last_stage = None
        while not has_started and not timeout_exceeded:
            resp = self._get_inference_progress()
            stage = resp.get("stage")
            if stage != last_stage:
                logger.info(stage)
                last_stage = stage
            has_started = stage not in ["preparing", "preprocess", None]
            has_started = has_started or bool(resp.get("result")) or resp["progress"]["total"] != 1
            self._update_progress_from_response(tqdm, resp)
            if not has_started:
                time.sleep(delay)
            timeout_exceeded = timeout and time.time() - t0 > timeout
        if timeout_exceeded:
            self.stop()
            raise Timeout("Timeout exceeded. The server didn't start the inference")
        return resp, has_started

    def _wait_for_pending_results(self, delay=1, timeout=600, tqdm: tqdm = None) -> List[dict]:
        logger.debug("waiting pending results...")
        has_results = False
        timeout_exceeded = False
        t0 = time.monotonic()
        while not has_results and not timeout_exceeded:
            resp = self._pop_pending_results()
            self._update_progress_from_response(tqdm, resp)
            pending_results = resp["pending_results"]
            exception_json = resp["exception"]
            if exception_json:
                exception_str = f"{exception_json['type']}: {exception_json['message']}"
                raise RuntimeError(f"Inference Error: {exception_str}")
            has_results = bool(pending_results)
            if resp.get("finished", False):
                break
            if not has_results:
                time.sleep(delay)
            timeout_exceeded = timeout and time.monotonic() - t0 > timeout
        if timeout_exceeded:
            self.stop()
            raise Timeout("Timeout exceeded. Pending results not received from the server.")
        return pending_results

    def _start_inference(self, method, **kwargs):
        if self.inference_request_uuid:
            raise RuntimeError(
                "Inference is already running. Please stop it before starting a new one."
            )
        resp = self._post(method, **kwargs).json()
        self.inference_request_uuid = resp["inference_request_uuid"]
        try:
            resp, has_started = self._wait_for_inference_start(tqdm=self.tqdm)
        except:
            self.stop()
            raise
        logger.info(
            "Inference has started:",
            extra={"inference_request_uuid": resp.get("inference_request_uuid")},
        )
        frame_iterator = self.Iterator(resp["progress"]["total"], self, tqdm=self.tqdm)
        return frame_iterator

    def _predict_images(self, images: List, **kwargs: dict):
        if isinstance(images[0], bytes):
            f = self._predict_images_bytes
        elif isinstance(images[0], (str, PathLike)):
            f = self._predict_images_paths
        elif isinstance(images[0], np.ndarray):
            f = self._predict_images_nps
        elif isinstance(images[0], int):
            f = self._predict_images_ids
        else:
            raise ValueError(f"Unsupported input type '{type(images[0])}'.")
        kwargs = get_valid_kwargs(kwargs, f, exclude=["images"])
        return f(images, **kwargs)

    def _predict_images_bytes(self, images: List[bytes], batch_size: int = None):
        files = [
            ("files", (f"image_{i}.png", image, "image/png")) for i, image in enumerate(images)
        ]
        state = self._get_json_body()["state"]
        if batch_size is not None:
            state["batch_size"] = batch_size
        method = "inference_batch_async"
        uploads = files + [("state", (None, json.dumps(state), "text/plain"))]
        return self._start_inference(method, files=uploads)

    def _predict_images_paths(self, images: List, batch_size: int = None):
        files = []
        try:
            files = [("files", open(f, "rb")) for f in images]
            state = self._get_json_body()["state"]
            if batch_size is not None:
                state["batch_size"] = batch_size
            method = "inference_batch_async"
            uploads = files + [("state", (None, json.dumps(state), "text/plain"))]
            return self._start_inference(method, files=uploads)
        finally:
            for _, f in files:
                f.close()

    def _predict_images_nps(self, images: List, batch_size: int = None):
        images = [write_bytes(image, ".png") for image in images]
        return self._predict_images_bytes(images, batch_size=batch_size)

    def _predict_images_ids(
        self,
        images: List[int],
        batch_size: int = None,
        upload_mode: str = None,
        output_project_id: int = None,
    ):
        method = "inference_batch_ids_async"
        json_body = self._get_json_body()
        state = json_body["state"]
        state["images_ids"] = images
        if batch_size is not None:
            state["batch_size"] = batch_size
        if upload_mode is not None:
            state["upload_mode"] = upload_mode
        if output_project_id is not None:
            state["output_project_id"] = output_project_id
        return self._start_inference(method, json=json_body)

    def _predict_videos(
        self,
        videos: Union[List[int], List[str], List[PathLike]],
        start_frame: int = None,
        num_frames: int = None,
        stride=None,
        end_frame=None,
        duration=None,
        direction: Literal["forward", "backward"] = None,
        tracker: Literal["botsort"] = None,
        tracker_settings: dict = None,
        batch_size: int = None,
    ):
        if len(videos) != 1:
            raise ValueError("Only one video can be processed at a time.")
        json_body = self._get_json_body()
        state = json_body["state"]
        for key, value in (
            ("start_frame", start_frame),
            ("num_frames", num_frames),
            ("stride", stride),
            ("end_frame", end_frame),
            ("duration", duration),
            ("direction", direction),
            ("tracker", tracker),
            ("tracker_settings", tracker_settings), 
            ("batch_size", batch_size),
        ):
            if value is not None:
                state[key] = value
        if isinstance(videos[0], int):
            method = "inference_video_id_async"
            state["video_id"] = videos[0]
            return self._start_inference(method, json=json_body)
        elif isinstance(videos[0], (str, PathLike)):
            video_path = videos[0]
            files = []
            try:
                method = "inference_video_async"
                files.append((Path(video_path).name, open(video_path, "rb"), "video/*"))
                fields = {
                    "files": files[-1],
                    "state": json.dumps(state),
                }
                encoder = MultipartEncoder(fields)
                if self.tqdm is not None:

                    bytes_read = 0
                    def _callback(monitor):
                        nonlocal bytes_read
                        self.tqdm.update(monitor.bytes_read - bytes_read)
                        bytes_read = monitor.bytes_read

                    video_size = get_file_size(video_path)
                    self._update_progress(self.tqdm, "Uploading video", 0, video_size, is_size=True)
                    encoder = MultipartEncoderMonitor(encoder, _callback)

                return self._start_inference(
                    method, data=encoder, headers={"Content-Type": encoder.content_type}
                )
            finally:
                for _, f, _ in files:
                    f.close()
        else:
            raise ValueError(
                f"Unsupported input type '{type(videos[0])}'. Supported types are: int, str, PathLike."
            )

    def _predict_projects(
        self,
        project_ids: List[int],
        dataset_ids: List[int] = None,
        batch_size: int = None,
        upload_mode: str = None,
        iou_merge_threshold: float = None,
        cache_project_on_model: bool = None,
        output_project_id: int = None,
    ):
        if len(project_ids) != 1:
            raise ValueError("Only one project can be processed at a time.")
        method = "inference_project_id_async"
        json_body = self._get_json_body()
        state = json_body["state"]
        state["project_id"] = project_ids[0]
        if dataset_ids is not None:
            state["dataset_ids"] = dataset_ids
        if batch_size is not None:
            state["batch_size"] = batch_size
        if upload_mode is not None:
            state["upload_mode"] = upload_mode
        if iou_merge_threshold is not None:
            state["iou_merge_threshold"] = iou_merge_threshold
        if cache_project_on_model is not None:
            state["cache_project_on_model"] = cache_project_on_model
        if output_project_id is not None:
            state["output_project_id"] = output_project_id
        return self._start_inference(method, json=json_body)

    def _predict_datasets(
        self,
        dataset_ids: List[int],
        batch_size: int = None,
        upload_mode: str = None,
        iou_merge_threshold: float = None,
        cache_datasets_on_model: bool = None,
    ):
        if self.api is None:
            raise ValueError("Api is required to use this method.")
        dataset_infos = [self.api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
        if len(set([info.project_id for info in dataset_infos])) > 1:
            raise ValueError("All datasets must belong to the same project.")
        return self._predict_projects(
            [dataset_infos[0].project_id],
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            upload_mode=upload_mode,
            iou_merge_threshold=iou_merge_threshold,
            cache_project_on_model=cache_datasets_on_model,
        )
