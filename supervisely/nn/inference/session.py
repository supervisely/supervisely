import json
import time
import requests
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
import yaml
from requests import Timeout

import supervisely as sly
from supervisely.io.network_exceptions import process_requests_exception
from supervisely.sly_logger import logger


### InferenceSession
class Session:
    def __init__(
        self,
        api: sly.Api,
        task_id: int = None,
        session_token: str = None,
        inference_settings: Union[dict, str] = None,
    ):
        # inference_settings is a dict with settings or path to .yml file
        if (task_id is None and session_token is None) or (
            task_id is not None and session_token is not None
        ):
            raise RuntimeError(
                "exactly one of `task_id` or `session_token` has to be passed as parameter."
            )

        self.api = api
        self._task_id = int(task_id)
        self._session_token = str(
            session_token or api.task.get_info_by_id(task_id)["meta"]["sessionToken"]
        )
        self.inference_settings = inference_settings or {}
        if isinstance(self.inference_settings, str):
            yaml_path = self.inference_settings
            with open(yaml_path, "r") as f:
                self.inference_settings: dict = yaml.safe_load(f)
        self._base_url = f"{self.api.server_address}/net/{self._session_token}"
        self._session_info = None
        self._default_inference_settings = None
        self._model_project_meta = None
        self._async_inference_uuid = None
        self._stop_async_inference_flag = False

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def session_token(self) -> str:
        return self._session_token

    def get_session_info(self):
        if self._session_info is None:
            self._session_info = self._get_from_endpoint("get_session_info")
        return self._session_info

    def get_default_inference_settings(self):
        if self._default_inference_settings is None:
            resp = self._get_from_endpoint("get_custom_inference_settings")
            settings = resp["settings"]
            if isinstance(settings, str):
                settings = yaml.safe_load(settings)
            self._default_inference_settings = settings
        return self._default_inference_settings

    def get_model_project_meta(self):
        if self._model_project_meta is None:
            meta_json = self._get_from_endpoint("get_output_classes_and_tags")
            self._model_project_meta = sly.ProjectMeta.from_json(meta_json)
        return self._model_project_meta

    def update_inference_settings(self, **inference_settings):
        self.inference_settings.update(inference_settings)
        return self.inference_settings

    def infer_image_id(self, image_id: int):
        endpoint = "inference_image_id"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["image_id"] = image_id
        resp = self._post(url, json=json_body)
        return resp.json()

    def infer_image_ids(self, image_ids: List[int]):
        endpoint = "inference_batch_ids"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["batch_ids"] = image_ids
        resp = self._post(url, json=json_body)
        return resp.json()

    def infer_image_url(self, url: str):
        endpoint = "inference_image_url"
        endpoint_url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        json_body["state"]["image_url"] = url
        resp = self._post(endpoint_url, json=json_body)
        return resp.json()

    def infer_image_path(self, image_path: str):
        endpoint = "inference_image"
        url = f"{self._base_url}/{endpoint}"
        opened_file = open(image_path, "rb")
        settings_json = json.dumps({"settings": self.inference_settings})
        uploads = [("files", opened_file), ("settings", (None, settings_json, "text/plain"))]
        resp = self._post(url, files=uploads)
        opened_file.close()
        return resp.json()

    def infer_image_paths(self, image_paths: List[str]):
        endpoint = "inference_batch"
        url = f"{self._base_url}/{endpoint}"
        files = [("files", open(f, "rb")) for f in image_paths]
        settings_json = json.dumps({"settings": self.inference_settings})
        uploads = files + [("settings", (None, settings_json, "text/plain"))]
        resp = self._post(url, files=uploads)
        for _, f in files:
            f.close()
        return resp.json()

    def infer_video_id(
        self,
        video_id: int,
        startFrameIndex: int = None,
        framesCount: int = None,
        framesDirection: Literal["forward", "backward"] = None,
    ):
        endpoint = "inference_video_id"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["videoId"] = video_id
        state.update(
            self._collect_state_for_infer_video(startFrameIndex, framesCount, framesDirection)
        )
        resp = self._post(url, json=json_body)
        return resp.json()

    def infer_video_id_async(
        self,
        video_id: int,
        startFrameIndex: int = None,
        framesCount: int = None,
        framesDirection: Literal["forward", "backward"] = None,
    ):
        if self._async_inference_uuid:
            raise RuntimeError("Can processing only one inference at time.")
        endpoint = "inference_video_id_async"
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        state = json_body["state"]
        state["videoId"] = video_id
        state.update(
            self._collect_state_for_infer_video(startFrameIndex, framesCount, framesDirection)
        )
        resp = self._post(url, json=json_body).json()
        self._async_inference_uuid = resp["inference_request_uuid"]
        self._stop_async_inference_flag = False
        resp, has_started = self._wait_for_async_inference_start()
        logger.info("Inference has started:", extra=resp)
        frame_iterator = AsyncInferenceIterator(resp["progress"]["total"], self)
        return frame_iterator

    def stop_async_inference(self):
        endpoint = "stop_inference"
        resp = self._get_from_endpoint_for_async_inference(endpoint)
        self._stop_async_inference_flag = True
        logger.info("Inference will be stopped on the server")
        return resp

    def _get_inference_progress(self):
        endpoint = "get_inference_progress"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _pop_pending_results(self):
        endpoint = "pop_inference_results"
        return self._get_from_endpoint_for_async_inference(endpoint)

    def _wait_for_async_inference_start(self, delay=1, timeout=None):
        logger.info("The video is preparing on the server, this may take a while...")
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

    def _wait_for_new_pending_results(self, delay=1, timeout=600):
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
        if not pending_results and resp["is_inferring"]:
            logger.warn(
                "The model is inferring yet, but new pending results have not received from the serving app. "
                "This may lead to not all samples will be inferred."
            )
        return pending_results

    def _on_async_inference_end(self):
        logger.debug("callback: on_async_inference_end")
        self._async_inference_uuid = None

    def _post(self, retries=5, *args, **kwargs):
        url = args[0]
        method = "/net"
        for retry_idx in range(len(retries)):
            try:
                response = requests.post(*args, **kwargs)
                if response.status_code != requests.codes.ok:
                    sly.Api._raise_for_status(response)
                break  # break if no exceptions
            except requests.RequestException as exc:
                process_requests_exception(
                    logger,
                    exc,
                    method,
                    url,
                    verbose=True,
                    swallow_exc=True,
                    sleep_sec=3,
                    response=response,
                    retry_info={"retry_idx": retry_idx + 1, "retry_limit": retries},
                )
        return response

    def _get_from_endpoint(self, endpoint):
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body()
        resp = self._post(url, json=json_body)
        return resp.json()

    def _get_from_endpoint_for_async_inference(self, endpoint):
        url = f"{self._base_url}/{endpoint}"
        json_body = self._get_default_json_body_for_async_inference()
        resp = self._post(url, json=json_body)
        return resp.json()

    def _collect_state_for_infer_video(
        self,
        startFrameIndex: int = None,
        framesCount: int = None,
        framesDirection: Literal["forward", "backward"] = None,
    ):
        state = {}
        if startFrameIndex is not None:
            state["startFrameIndex"] = startFrameIndex
        if framesCount is not None:
            state["framesCount"] = framesCount
        if framesDirection is not None:
            state["framesDirection"] = framesDirection
        return state

    def _get_default_json_body(self):
        return {
            "state": {"settings": self.inference_settings},
            "context": {},
            "api_token": self.api.token,
        }

    def _get_default_json_body_for_async_inference(self):
        json_body = self._get_default_json_body()
        json_body["state"]["inference_request_uuid"] = self._async_inference_uuid
        return json_body


class AsyncInferenceIterator:
    def __init__(self, total, nn_api: Session) -> None:
        self.total = total
        self.nn_api = nn_api
        self.results_queue = []

    def __len__(self):
        return self.total

    def __iter__(self):
        return self

    def __next__(self):
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
        return self.results_queue.pop(0)
