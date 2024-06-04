import json
import concurrent
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, dirname, join
from typing import List, Literal, NamedTuple, Optional
from time import time
import requests

from supervisely import logger
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo
from supervisely.io.fs import silent_remove
from supervisely.io.json import dump_json_file


class CheckpointInfo(NamedTuple):
    """
    CheckpointInfo
    """

    app_name: str
    session_id: int
    session_path: str
    session_link: str
    task_type: str
    training_project_name: str
    checkpoints: List[FileInfo]
    config: str = None


class Checkpoint:
    SUPPORTED_FRAMEWORKS = [
        "yolov5",
        "yolov5_v2",
        "yolov8",
        "detectron2",
        "mmdetection",
        "mmdetection3",
        "mmsegmentation",
        "mmclassification",
        "ritm",
        "unet",
        "hrda",
    ]

    MODEL_DIRS = {
        "yolov5": "/yolov5_train",
        "yolov5_v2": "/yolov5_2.0_train",
        "yolov8": "/yolov8_train",
        "detectron2": "/detectron2",
        "mmdetection": "/mmdetection",
        "mmdetection3": "/mmdetection-3",
        "mmsegmentation": "/mmsegmentation",
        "mmclassification": "/mmclassification",
        "ritm": "/RITM_training",
        "unet": "/unet",
        "hrda": "/hrda",
    }

    APP_NAME = {
        "yolov5": "Train YOLOv5",
        "yolov5_v2": "Train YOLOv5 v2.0",
        "yolov8": "Train YOLOv8",
        "detectron2": "Train Detectron2",
        "mmdetection": "Train MMDetection",
        "mmdetection3": "Train MMDetection 3.0",
        "mmsegmentation": "Train MMsegmentation",
        "mmclassification": "Train MMclassification",
        "ritm": "Train RITM",
        "unet": "Train UNet",
        "hrda": "Train HRDA",
    }

    def __init__(self, team_id: int):
        self._api: Api = Api.from_env()
        self._team_id: int = team_id
        self._metadata_file_name = "sly_metadata.json"
        self._http_session = requests.Session()
        

    @property
    def models_dir(self) -> str:
        raise NotImplementedError

    def _sort_checkpoints(
        self, checkpoints: List[FileInfo], sort: Literal["desc", "asc"] = "desc"
    ) -> List[CheckpointInfo]:
        start_sort_time = time()
        checkpoints_with_ids = [(c.session_id, c) for c in checkpoints]
        if sort == "desc":
            checkpoints_with_ids.sort(reverse=True)
        elif sort == "asc":
            checkpoints_with_ids.sort()
        checkpoints = [c for _, c in checkpoints_with_ids]
        end_sort_time = time()
        logger.info(f"Sort time: '{format(end_sort_time - start_sort_time, '.6f')}' sec")
        return checkpoints

    def _add_sly_metadata(
        self,
        app_name: str,
        session_id: str,
        session_path: str,
        weights_path: str,
        weights_ext: str,
        training_project_name: str,
        task_type: str = None,
        config_path: str = None,
    ):
        def _get_checkpoints_infos(weights_path) -> List[FileInfo]:
            return [
                file
                for file in self._api.file.list(
                    self._team_id,
                    weights_path,
                    recursive=False,
                    return_type="fileinfo",
                )
                if file.name.endswith(weights_ext)
            ]

        def _upload_metadata(json_data: dict) -> None:
            json_data_path = self._metadata_file_name
            dump_json_file(json_data, json_data_path)
            self._api.file.upload(
                self._team_id,
                json_data_path,
                f"{session_path}/{self._metadata_file_name}",
            )
            silent_remove(json_data_path)

        checkpoints_infos = _get_checkpoints_infos(weights_path)
        if len(checkpoints_infos) == 0:
            logger.info(f"No checkpoints found in '{session_path}'")
            return None

        logger.info(f"Generating '{self._metadata_file_name}' for {session_path}")
        if is_development():
            session_link = abs_url(f"/apps/sessions/{session_id}")
        else:
            session_link = f"/apps/sessions/{session_id}"

        json_data = {
            "app_name": app_name,
            "session_id": session_id,
            "session_path": session_path,
            "session_link": session_link,
            "task_type": task_type,
            "training_project_name": training_project_name,
            "checkpoints": checkpoints_infos,
        }

        if config_path is not None:
            json_data["config"] = config_path

        _upload_metadata(json_data)
        return json_data


    # def _fetch_json_from_url(self, metadata_url: str):
    #     with ThreadPoolExecutor(max_workers=10) as executor:
    #         future_to_url = {executor.submit(self._fetch_single_json, url): url for url in metadata_url}
    #         for future in concurrent.futures.as_completed(future_to_url):
    #             url = future_to_url[future]
    #             try:
    #                 data = future.result()
    #             except Exception as exc:
    #                 logger.debug(f"Failed to fetch model metadata from '{url}'")
    #             else:
    #                 return data

    # def _fetch_single_json(self, url):
    #     try:
    #         response = requests.get(url)
    #         response.raise_for_status()
    #         response_json = response.json()
    #         checkpoints = response_json.get("checkpoints", [])
    #         file_infos = [FileInfo(*checkpoint) for checkpoint in checkpoints]
    #         response_json["checkpoints"] = file_infos
    #         return response_json
    #     except:
    #         logger.debug(f"Failed to fetch model metadata from '{url}'")
    #         return None

    def _fetch_json_from_url(self, metadata_url: str):
        try:
            response = self._http_session.get(metadata_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch model metadata from '{metadata_url}': {e}")
            return None

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to decode JSON from '{metadata_url}': {e}")
            return None

        checkpoints = response_json.get("checkpoints", [])
        file_infos = [FileInfo(*checkpoint) for checkpoint in checkpoints]
        response_json["checkpoints"] = file_infos

        return response_json

    def get_list(
        self,
        framework: str,
        progress=None,
    ) -> List[CheckpointInfo]:
        framework = framework.lower()
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {framework}")

        get_list_function = self.SUPPORTED_FRAMEWORKS[framework].get_list
        checkpoints = get_list_function(self._api, self.team_id)
        return checkpoints
