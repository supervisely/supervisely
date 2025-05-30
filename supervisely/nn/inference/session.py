import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal

import yaml
from requests import HTTPError, Timeout

import supervisely as sly
from supervisely.convert.image.sly.sly_image_helper import get_meta_from_annotation
from supervisely.io.network_exceptions import process_requests_exception
from supervisely.nn.utils import DeployInfo
from supervisely.sly_logger import logger


class SessionJSON:
    def __init__(
        self,
        api: sly.Api,
        task_id: int = None,
        session_url: str = None,
        inference_settings: Union[dict, str] = None,
    ):
        """
        A convenient class for inference of deployed models.
        This class will return raw JSON predictions (dict).
        If you want to work with `sly.Annotation` format, please use `sly.nn.inference.Session` class.

        You need first to serve NN model you want and get its `task_id`.
        The `task_id` can be obtained from the Supervisely platform, going to `START` -> `App sessions` page.

        Note: Either a `task_id` or a `session_url` has to be passed as a parameter (not both).

        :param api: initialized :class:`sly.Api` object.
        :type api: sly.Api
        :param task_id: the task_id of a served model in the Supervisely platform. If None, the `session_url` will be used instead, defaults to None
        :type task_id: int, optional
        :param session_url: the url for direct connection to the served model. If None, the `task_id` will be used instead, defaults to None
        :type session_url: str, optional
        :param inference_settings: a dict or a path to YAML file with settings, defaults to None
        :type inference_settings: Union[dict, str], optional


        :Usage example:
         .. code-block:: python
            task_id = 27001
            session = sly.nn.inference.SessionJSON(
                api,
                task_id=task_id,
            )
            print(session.get_session_info())

            image_id = 17551748
            pred = session.inference_image_id(image_id)
            predicted_annotation = sly.Annotation.from_json(pred["annotation"], model_meta)

        """
        assert not (
            task_id is None and session_url is None
        ), "Either `task_id` or `session_url` must be passed."
        assert (
            task_id is None or session_url is None
        ), "Either `task_id` or `session_url` must be passed (not both)."

        self.api = api
        self._task_id = task_id

        if task_id is not None:
            try:
                # TODO: api.task.get_info_by_id get stuck if the task_id isn't exists on the platform
                # https://github.com/supervisely/issues/issues/1873
                task_info = api.task.get_info_by_id(task_id)
            except HTTPError:
                raise ValueError(
                    f"Can't connect to the model. Check if the task_id {task_id} is correct."
                )
        if task_id is not None:
            self._base_url = f'{self.api.server_address}/net/{task_info["meta"]["sessionToken"]}'
        else:
            self._base_url = session_url
        self.inference_settings = {}
        self._session_info = None
        self._default_inference_settings = None
        self._model_meta = None
        self._async_inference_uuid = None
        self._stop_async_inference_flag = False
        self.inference_result = None

        if inference_settings is not None:
            self.set_inference_settings(inference_settings)

        # Check connection:
        try:
            self.get_session_info()
        except HTTPError:
            if task_id is not None:
                raise ValueError(
                    f"Can't connect to the model. Check if the task_id {self._task_id} is correct."
                )
            else:
                raise ValueError(
                    f'Can\'t connect to the model. Check if the session_url "{session_url}" is correct.'
                )

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def base_url(self) -> str:
        return self._base_url

    def get_session_info(self) -> Dict[str, Any]:
        self._session_info = self._get_from_endpoint("get_session_info")
        return self._session_info

    def get_human_readable_info(self, replace_none_with: Optional[str] = None):
        hr_info = {}
        info = self.get_session_info()

        for name, data in info.items():
            hr_name = name.replace("_", " ").capitalize()
            if data is None:
                hr_info[hr_name] = replace_none_with
            else:
                hr_info[hr_name] = data

        return hr_info

    def get_model_meta(self) -> Dict[str, Any]:
        meta_json = self._get_from_endpoint("get_output_classes_and_tags")
        self._model_meta = meta_json
        return self._model_meta

    def get_deploy_info(self) -> Dict[str, Any]:
        return self._get_from_endpoint("get_deploy_info")

    def get_default_inference_settings(self) -> Dict[str, Any]:
        if self._default_inference_settings is None:
            resp = self._get_from_endpoint("get_custom_inference_settings")
            settings = resp["settings"]
            if isinstance(settings, str):
                settings = yaml.safe_load(settings)
            self._default_inference_settings = settings
        return self._default_inference_settings

    def update_inference_settings(self, **inference_settings) -> Dict[str, Any]:
        self._validate_new_inference_settings(inference_settings)
        self.inference_settings.update(inference_settings)
        return self.inference_settings

    def set_inference_settings(self, inference_settings) -> Dict[str, Any]:
        self._set_inference_settings_dict_or_yaml(inference_settings)
        return self.inference_settings

    def _set_inference_settings_dict_or_yaml(self, dict_or_yaml_path) -> None:
        if isinstance(dict_or_yaml_path, str):
            with open(dict_or_yaml_path, "r") as f:
                new_settings = yaml.safe_load(f)
        elif isinstance(dict_or_yaml_path, dict):
            new_settings = dict_or_yaml_path
        else:
            raise ValueError(
                "The `inference_settings` parameter must be either a dict or a path to a YAML file."
            )
        self._validate_new_inference_settings(new_settings)
        self.inference_settings = new_settings

    def _validate_new_inference_settings(self, new_settings: dict) -> None:
        default_settings = self.get_default_inference_settings()
        for key, value in new_settings.items():
            if key not in default_settings and key != "classes":
                acceptable_keys = ", ".join(default_settings.keys()) + ", 'classes'"
                raise ValueError(
                    f"Key '{key}' is not acceptable. Acceptable keys are: {acceptable_keys}"
                )

    def inference_image_id(self, image_id: int, upload=False) -> Dict[str, Any]:
        endpoint = "inference_image_id"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["image_id"] = image_id
        json_body["state"]["upload"] = upload
        resp = self._post(url, json=json_body)
        return resp.json()

    def inference_image_ids(self, image_ids: List[int]) -> List[Dict[str, Any]]:
        endpoint = "inference_batch_ids"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["batch_ids"] = image_ids
        resp = self._post(url, json=json_body)
        return resp.json()

    def inference_image_ids_async(
        self,
        image_ids: List[int],
        output_project_id: int = None,
        batch_size: int = None,
        process_fn=None,
    ) -> Iterator:
        if self._async_inference_uuid:
            logger.info(
                "Trying to run a new inference while `_async_inference_uuid` already exists. Stopping the old one..."
            )
            try:
                self.stop_async_inference()
                self._on_async_inference_end()
            except Exception as exc:
                logger.error(f"An error has occurred while stopping the previous inference. {exc}")
        endpoint = "inference_batch_ids_async"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["images_ids"] = image_ids
        state["output_project_id"] = output_project_id
        state["batch_size"] = batch_size
        resp = self._post(url, json=json_body).json()
        self._async_inference_uuid = resp["inference_request_uuid"]
        self._stop_async_inference_flag = False

        logger.info("Inference has started:", extra={"response": resp})
        resp, has_started = self._wait_for_async_inference_start()
        frame_iterator = AsyncInferenceIterator(
            resp["progress"]["total"], self, process_fn=process_fn
        )
        return frame_iterator

    def inference_image_url(self, url: str) -> Dict[str, Any]:
        endpoint = "inference_image_url"
        endpoint_url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["image_url"] = url
        resp = self._post(endpoint_url, json=json_body)
        return resp.json()

    def inference_image_path(self, image_path: str) -> Dict[str, Any]:
        endpoint = "inference_image"
        url = f"{self._base_url}/{endpoint}"
        opened_file = open(image_path, "rb")
        settings_json = json.dumps({"settings": self.inference_settings})
        uploads = [
            ("files", opened_file),
            ("settings", (None, settings_json, "text/plain")),
        ]
        resp = self._post(url, files=uploads)
        opened_file.close()
        return resp.json()

    def inference_image_paths(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        endpoint = "inference_batch"
        url = f"{self._base_url}/{endpoint}"
        files = [("files", open(f, "rb")) for f in image_paths]
        settings_json = json.dumps({"settings": self.inference_settings})
        uploads = files + [("settings", (None, settings_json, "text/plain"))]
        resp = self._post(url, files=uploads)
        for _, f in files:
            f.close()
        return resp.json()

    def inference_video_id(
        self,
        video_id: int,
        start_frame_index: int = None,
        frames_count: int = None,
        frames_direction: Literal["forward", "backward"] = None,
        tracker: Literal["bot", "deepsort"] = None,
        batch_size: int = None,
    ) -> Dict[str, Any]:
        endpoint = "inference_video_id"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["videoId"] = video_id
        state["batch_size"] = batch_size
        state.update(
            self._collect_state_for_infer_video(start_frame_index, frames_count, frames_direction)
        )
        state["tracker"] = tracker
        resp = self._post(url, json=json_body)
        return resp.json()

    def inference_video_id_async(
        self,
        video_id: int,
        start_frame_index: int = None,
        frames_count: int = None,
        frames_direction: Literal["forward", "backward"] = None,
        process_fn=None,
        preparing_cb=None,
        tracker: Literal["bot", "deepsort"] = None,
        batch_size: int = None,
    ) -> Iterator:
        if self._async_inference_uuid:
            logger.info(
                "Trying to run a new inference while `_async_inference_uuid` already exists. Stopping the old one..."
            )
            try:
                self.stop_async_inference()
                self._on_async_inference_end()
            except Exception as exc:
                logger.error(f"An error has occurred while stopping the previous inference. {exc}")
        endpoint = "inference_video_id_async"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["videoId"] = video_id
        state["batch_size"] = batch_size
        state.update(
            self._collect_state_for_infer_video(start_frame_index, frames_count, frames_direction)
        )
        state["tracker"] = tracker
        resp = self._post(url, json=json_body).json()
        self._async_inference_uuid = resp["inference_request_uuid"]
        self._stop_async_inference_flag = False

        current = 0
        prev_current = 0
        if preparing_cb:
            # wait for inference status
            resp = self._get_preparing_progress()
            while resp.get("status") is None:
                time.sleep(2)
                resp = self._get_preparing_progress()

            if resp["status"] == "download_video":
                progress_widget = preparing_cb(
                    message="Downloading Video",
                    total=resp["total"],
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            else:
                progress_widget = preparing_cb(message="Downloading Frames", total=resp["total"])

            while resp["status"] in ["download_video", "download_frames"]:
                resp = self._get_preparing_progress()
                current = resp["current"]
                progress_widget.update(current - prev_current)
                prev_current = current

            resp = self._get_preparing_progress()
            if resp["status"] == "cut":
                progress_widget = preparing_cb(message="Cutting frames", total=resp["total"])
            while resp["status"] == "cut":
                resp = self._get_preparing_progress()
                current = resp["current"]
                progress_widget.update(current - prev_current)
                prev_current = current

        logger.info("Inference has started:", extra={"response": resp})
        resp, has_started = self._wait_for_async_inference_start()
        frame_iterator = AsyncInferenceIterator(
            resp["progress"]["total"], self, process_fn=process_fn
        )
        return frame_iterator

    def inference_project_id_async(
        self,
        project_id: int,
        dataset_ids: List[int] = None,
        output_project_id: int = None,
        cache_project_on_model: bool = False,
        batch_size: int = None,
        process_fn=None,
    ):
        if self._async_inference_uuid:
            logger.info(
                "Trying to run a new inference while `_async_inference_uuid` already exists. Stopping the old one..."
            )
            try:
                self.stop_async_inference()
                self._on_async_inference_end()
            except Exception as exc:
                logger.error(f"An error has occurred while stopping the previous inference. {exc}")
        endpoint = "inference_project_id_async"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["projectId"] = project_id
        state["output_project_id"] = output_project_id
        state["cache_project_on_model"] = cache_project_on_model
        state["dataset_ids"] = dataset_ids
        state["batch_size"] = batch_size
        resp = self._post(url, json=json_body).json()
        self._async_inference_uuid = resp["inference_request_uuid"]
        self._stop_async_inference_flag = False

        logger.info("Inference has started:", extra={"response": resp})
        resp, has_started = self._wait_for_async_inference_start()
        frame_iterator = AsyncInferenceIterator(
            resp["progress"]["total"], self, process_fn=process_fn
        )
        return frame_iterator

    def run_speedtest(
        self,
        project_id: int,
        batch_size: int,
        num_iterations: int = 100,
        num_warmup: int = 3,
        dataset_ids: List[int] = None,
        cache_project_on_model: bool = False,
        preparing_cb=None,
    ):
        if self._async_inference_uuid:
            logger.info(
                "Trying to run a new inference while `_async_inference_uuid` already exists. Stopping the old one..."
            )
            try:
                self.stop_async_inference()
                self._on_async_inference_end()
            except Exception as exc:
                logger.error(f"An error has occurred while stopping the previous inference. {exc}")
        endpoint = "run_speedtest"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        params = {
            "projectId": project_id,
            "cache_project_on_model": cache_project_on_model,
            "dataset_ids": dataset_ids,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "num_warmup": num_warmup,
        }
        state.update(params)
        resp = self._post(url, json=json_body).json()
        self._async_inference_uuid = resp["inference_request_uuid"]
        self._stop_async_inference_flag = False

        current = 0
        prev_current = 0
        if preparing_cb:
            # wait for inference status
            resp = self._get_preparing_progress()
            awaiting_preparing_progress = 0
            break_flag = False
            while resp.get("status") is None:
                time.sleep(1)
                awaiting_preparing_progress += 1
                if awaiting_preparing_progress > 30:
                    break_flag = True
                resp = self._get_preparing_progress()
            if break_flag:
                logger.warning(
                    "Unable to get preparing progress. Continue without prepaing progress status."
                )
            if not break_flag:
                if resp["status"] == "download_info":
                    progress_widget = preparing_cb(
                        message="Downloading infos", total=resp["total"], unit="it"
                    )
                while resp["status"] == "download_info":
                    current = resp["current"]
                    # pylint: disable=possibly-used-before-assignment
                    progress_widget.update(current - prev_current)
                    prev_current = current
                    resp = self._get_preparing_progress()

                if resp["status"] == "download_project":
                    progress_widget = preparing_cb(message="Download project", total=resp["total"])
                while resp["status"] == "download_project":
                    current = resp["current"]
                    progress_widget.update(current - prev_current)
                    prev_current = current
                    resp = self._get_preparing_progress()

                if resp["status"] == "warmup":
                    progress_widget = preparing_cb(message="Running warmup", total=resp["total"])
                while resp["status"] == "warmup":
                    current = resp["current"]
                    progress_widget.update(current - prev_current)
                    prev_current = current
                    resp = self._get_preparing_progress()

        logger.info("Inference has started:", extra={"response": resp})
        resp, has_started = self._wait_for_async_inference_start()
        frame_iterator = AsyncInferenceIterator(resp["progress"]["total"], self, process_fn=None)
        return frame_iterator

    def inference_project_id(
        self,
        project_id: int,
        dataset_ids: List[int] = None,
        output_project_id: int = None,
        cache_project_on_model: bool = False,
        batch_size: int = None,
    ):
        return [
            pred
            for pred in self.inference_project_id_async(
                project_id,
                dataset_ids,
                output_project_id,
                cache_project_on_model=cache_project_on_model,
                batch_size=batch_size,
                process_fn=None,
            )
        ]

    def stop_async_inference(self) -> Dict[str, Any]:
        endpoint = "stop_inference"
        resp = self._get_from_endpoint_for_async_inference(endpoint)
        self._stop_async_inference_flag = True
        logger.info("Inference will be stopped on the server")
        return resp

    def _get_inference_progress(self) -> Dict[str, Any]:
        endpoint = "get_inference_progress"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _get_preparing_progress(self) -> Dict[str, Any]:
        endpoint = "get_preparing_progress"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _pop_pending_results(self) -> Dict[str, Any]:
        endpoint = "pop_inference_results"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _clear_inference_request(self) -> Dict[str, Any]:
        endpoint = "clear_inference_request"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _wait_for_async_inference_start(self, delay=1, timeout=None) -> Tuple[dict, bool]:
        logger.info("Preparing data on the model, this may take a while...")
        has_started = False
        timeout_exceeded = False
        t0 = time.time()
        while not has_started and not timeout_exceeded:
            resp = self._get_inference_progress()
            has_started = bool(resp["result"]) or resp["progress"]["total"] != 1
            if not has_started:
                time.sleep(delay)
            timeout_exceeded = timeout and time.time() - t0 > timeout
        if timeout_exceeded:
            self.stop_async_inference()
            self._on_async_inference_end()
            raise Timeout("Timeout exceeded. The server didn't start the inference")
        return resp, has_started

    def _wait_for_new_pending_results(self, delay=1, timeout=600) -> List[dict]:
        logger.debug("waiting pending results...")
        has_results = False
        timeout_exceeded = False
        t0 = time.time()
        while not has_results and not timeout_exceeded:
            resp = self._pop_pending_results()
            pending_results = resp["pending_results"]
            has_results = bool(pending_results)
            if resp["is_inferring"] is False:
                break
            if not has_results:
                time.sleep(delay)
            timeout_exceeded = timeout and time.time() - t0 > timeout
        if timeout_exceeded:
            self.stop_async_inference()
            self._on_async_inference_end()
            raise Timeout("Timeout exceeded. Pending results not received from the server.")
        if len(pending_results) == 0 and resp["is_inferring"]:
            logger.warn(
                "The model is inferring yet, but new pending results have not received from the serving app. "
                "This may lead to not all samples will be inferred."
            )
        return pending_results

    def _on_async_inference_end(self):
        logger.debug("callback: on_async_inference_end()")
        try:
            try:
                self.inference_result = self._get_inference_result()
            except Exception:
                pass
            self._clear_inference_request()
        finally:
            self._async_inference_uuid = None

    def _post(self, *args, retries=5, **kwargs) -> requests.Response:
        retries = min(self.api.retry_count, retries)
        url = kwargs.get("url") or args[0]
        method = url[len(self._base_url) :]
        for retry_idx in range(retries):
            response = None
            try:
                response = requests.post(*args, **kwargs)
                if response.status_code != requests.codes.ok:  # pylint: disable=no-member
                    sly.Api._raise_for_status(response)
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

    def _get_from_endpoint(self, endpoint) -> Dict[str, Any]:
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        resp = self._post(url, json=json_body)
        return resp.json()

    def _get_from_endpoint_for_async_inference(self, endpoint) -> Dict[str, Any]:
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body_for_async_inference()
        resp = self._post(url, json=json_body)
        return resp.json()

    def _get_inference_result(self):
        url = f"{self._base_url}/get_inference_result"
        json_body = self._get_default_json_body_for_async_inference()
        resp = self._post(url, json=json_body)
        return resp.json()

    def _collect_state_for_infer_video(
        self,
        start_frame_index: int = None,
        frames_count: int = None,
        frames_direction: Literal["forward", "backward"] = None,
    ) -> Dict[str, Any]:
        state = {}
        if start_frame_index is not None:
            state["startFrameIndex"] = start_frame_index
        if frames_count is not None:
            state["framesCount"] = frames_count
        if frames_direction is not None:
            state["framesDirection"] = frames_direction
        return state

    def _get_default_json_body(self) -> Dict[str, Any]:
        return {
            "state": {"settings": self.inference_settings},
            "context": {},
            "api_token": self.api.token,
        }

    def _get_default_json_body_for_async_inference(self) -> Dict[str, Any]:
        json_body = self._get_default_json_body()
        json_body["state"]["inference_request_uuid"] = self._async_inference_uuid
        return json_body


class AsyncInferenceIterator:
    def __init__(self, total, nn_api: SessionJSON, process_fn=None):
        self.total = total
        self.nn_api = nn_api
        self.results_queue = []
        self.process_fn = process_fn

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            if self.nn_api._stop_async_inference_flag:
                raise StopIteration
            if not self.results_queue:
                pending_results = self.nn_api._wait_for_new_pending_results()
                self.results_queue += pending_results
            if not self.results_queue:
                raise StopIteration

        except StopIteration as stop_iteration:
            self.nn_api._on_async_inference_end()
            raise stop_iteration
        except Exception as ex:
            # Any exceptions will be handled here
            self.nn_api.stop_async_inference()
            self.nn_api._on_async_inference_end()
            raise ex

        pred = self.results_queue.pop(0)
        if self.process_fn is not None:
            return self.process_fn(pred)
        else:
            return pred


class Session(SessionJSON):
    def __init__(
        self,
        api: sly.Api,
        task_id: int = None,
        session_url: str = None,
        inference_settings: Union[dict, str] = None,
    ):
        """
        A convenient class for inference of deployed models.
        This class will return predictions in the `sly.Annotation` format.
        If you want to work with raw JSON dicts, please use `sly.nn.inference.SessionJSON` class.

        You need first to serve NN model you want and get its `task_id`.
        The `task_id` can be obtained from the Supervisely platform, going to `START` -> `App sessions` page.

        Note: Either a `task_id` or a `session_url` has to be passed as a parameter (not both).

        :param api: initialized :class:`sly.Api` object.
        :type api: sly.Api
        :param task_id: the task_id of a served model in the Supervisely platform. If None, the `session_url` will be used instead, defaults to None
        :type task_id: int, optional
        :param session_url: the url for direct connection to the served model. If None, the `task_id` will be used instead, defaults to None
        :type session_url: str, optional
        :param inference_settings: a dict or a path to YAML file with settings, defaults to None
        :type inference_settings: Union[dict, str], optional


        :Usage example:
         .. code-block:: python
            task_id = 27001
            session = sly.nn.inference.Session(
                api,
                task_id=task_id,
            )
            print(session.get_session_info())

            image_id = 17551748
            predicted_annotation = session.inference_image_id(image_id)

        """
        super().__init__(api, task_id, session_url, inference_settings)

    def is_model_deployed(self):
        is_deployed = self.api.task.send_request(self._task_id, "is_deployed", {})
        return is_deployed

    def get_model_meta(self) -> sly.ProjectMeta:
        model_meta_json = super().get_model_meta()
        model_meta = sly.ProjectMeta.from_json(model_meta_json)
        self._model_meta = model_meta
        return self._model_meta

    def get_deploy_info(self) -> DeployInfo:
        return DeployInfo(**super().get_deploy_info())

    def inference_image_id(self, image_id: int, upload=False) -> sly.Annotation:
        pred_json = super().inference_image_id(image_id, upload)
        pred_ann = self._convert_to_sly_annotation(pred_json)
        return pred_ann

    def inference_image_path(self, image_path: str) -> sly.Annotation:
        pred_json = super().inference_image_path(image_path)
        pred_ann = self._convert_to_sly_annotation(pred_json)
        return pred_ann

    def inference_image_url(self, url: str) -> sly.Annotation:
        pred_json = super().inference_image_url(url)
        pred_ann = self._convert_to_sly_annotation(pred_json)
        return pred_ann

    def inference_image_ids(self, image_ids: List[int]) -> List[sly.Annotation]:
        pred_list_raw = super().inference_image_ids(image_ids)
        predictions = self._convert_to_sly_annotation_batch(pred_list_raw)
        return predictions

    def inference_image_ids_async(
        self,
        image_ids: List[int],
        output_project_id: int = None,
        batch_size: int = None,
    ):
        frame_iterator = super().inference_image_ids_async(
            image_ids,
            output_project_id,
            batch_size=batch_size,
            process_fn=self._convert_to_sly_ann_info,
        )
        return frame_iterator

    def inference_image_paths(self, image_paths: List[str]) -> List[sly.Annotation]:
        pred_list_raw = super().inference_image_paths(image_paths)
        predictions = self._convert_to_sly_annotation_batch(pred_list_raw)
        return predictions

    def inference_video_id(
        self,
        video_id: int,
        start_frame_index: int = None,
        frames_count: int = None,
        frames_direction: Literal["forward", "backward"] = None,
        tracker: Literal["bot", "deepsort"] = None,
        batch_size: int = None,
    ) -> List[sly.Annotation]:
        pred_list_raw = super().inference_video_id(
            video_id, start_frame_index, frames_count, frames_direction, tracker, batch_size
        )
        pred_list_raw = pred_list_raw["ann"]
        predictions = self._convert_to_sly_annotation_batch(pred_list_raw)
        return predictions

    def inference_video_id_async(
        self,
        video_id: int,
        start_frame_index: int = None,
        frames_count: int = None,
        frames_direction: Literal["forward", "backward"] = None,
        tracker: Literal["bot", "deepsort"] = None,
        batch_size: int = None,
        preparing_cb=None,
    ) -> AsyncInferenceIterator:
        frame_iterator = super().inference_video_id_async(
            video_id,
            start_frame_index,
            frames_count,
            frames_direction,
            process_fn=self._convert_to_sly_annotation,
            tracker=tracker,
            batch_size=batch_size,
            preparing_cb=preparing_cb,
        )
        return frame_iterator

    def inference_project_id_async(
        self,
        project_id: int,
        dataset_ids: List[int] = None,
        output_project_id: int = None,
        cache_project_on_model: bool = False,
        batch_size: int = None,
    ):
        frame_iterator = super().inference_project_id_async(
            project_id,
            dataset_ids,
            output_project_id,
            cache_project_on_model=cache_project_on_model,
            batch_size=batch_size,
            process_fn=self._convert_to_sly_ann_info,
        )
        return frame_iterator

    def inference_project_id(
        self,
        project_id: int,
        dataset_ids: List[int] = None,
        output_project_id: int = None,
        cache_project_on_model: bool = False,
        batch_size: int = None,
    ):
        return [
            pred
            for pred in self.inference_project_id_async(
                project_id, dataset_ids, output_project_id, cache_project_on_model, batch_size
            )
        ]

    def _convert_to_sly_ann_info(self, pred_json: dict):
        image_id = pred_json["image_id"]
        image_name = pred_json["image_name"]
        annotation = pred_json["annotation"]
        return sly.api.annotation_api.AnnotationInfo(
            image_id=image_id,
            image_name=image_name,
            annotation=annotation,
            created_at=None,
            updated_at=None,
        )

    def _convert_to_sly_annotation(self, pred_json: dict):
        model_meta = self.get_model_meta()
        meta = get_meta_from_annotation(pred_json["annotation"], model_meta)
        pred_ann = sly.Annotation.from_json(pred_json["annotation"], meta)
        return pred_ann

    def _convert_to_sly_annotation_batch(self, pred_list_raw: List[dict]):
        meta = self.get_model_meta()
        for pred in pred_list_raw:
            meta = get_meta_from_annotation(pred["annotation"], meta)
        predictions = [sly.Annotation.from_json(pred["annotation"], meta) for pred in pred_list_raw]
        return predictions
