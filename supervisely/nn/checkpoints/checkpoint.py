from os.path import basename, dirname, join
from typing import List, Literal, NamedTuple, Optional

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

    @property
    def models_dir(self) -> str:
        raise NotImplementedError

    def _generate_sly_metadata(
        self,
        experiment_dir: str,
        weights_dir: str,
        weights_ext: str,
        training_app: str,
        task_type: str = None,
        config_file: str = None,
        project_name: str = None,
    ):
        logger.info(f"Generating '{self._metadata_file_name}' for {experiment_dir}")

        if training_app.startswith("Train YOLO"):
            task_id = basename(dirname(experiment_dir))
        else:
            task_id = experiment_dir.split("_")[0]

        if project_name is None:
            project_name = experiment_dir.split("_")[1]

        if is_development():
            session_link = abs_url(f"/apps/sessions/{task_id}")
        else:
            session_link = f"/apps/sessions/{task_id}"
        path_to_checkpoints = join(experiment_dir, weights_dir)
        checkpoints_infos = [
            file
            for file in self._api.file.list(
                self._team_id,
                path_to_checkpoints,
                recursive=False,
                return_type="fileinfo",
            )
            if file.name.endswith(weights_ext)
        ]

        config_url = None
        if config_file is not None:
            config_url = join(experiment_dir, config_file)
            if not self._api.file.exists(self._team_id, config_url):
                return None
            if training_app == "Train MMDetection 3.0":
                self._api.file.download(self._team_id, config_url, "model_config.txt")
                with open("model_config.txt", "r") as f:
                    lines = f.readlines()
                    project_line = lines[-1] if lines else None
                    start = project_line.find("'") + 1
                    end = project_line.find("'", start)
                    project_name = project_line[start:end]
                    task_type_line = lines[-3] if lines else None
                    start = task_type_line.find("'") + 1
                    end = task_type_line.find("'", start)
                    task_type = task_type_line[start:end].replace("_", " ")
                    f.close()
                silent_remove("model_config.txt")

        if len(checkpoints_infos) == 0:
            return None

        json_data = {
            "app_name": training_app,
            "session_id": task_id,
            "session_path": experiment_dir,
            "session_link": session_link,
            "task_type": task_type,
            "training_project_name": project_name,
            "checkpoints": checkpoints_infos,
        }

        if config_url is not None:
            json_data["config"] = config_url

        json_data_path = self._metadata_file_name
        dump_json_file(json_data, json_data_path)
        self._api.file.upload(
            self._team_id,
            json_data_path,
            f"{experiment_dir}/{self._metadata_file_name}",
        )
        silent_remove(json_data_path)
        return json_data

    def _fetch_json_from_url(self, url: str):
        metadata: FileInfo = self._api.file.get_info_by_path(self._team_id, url)
        if metadata is not None:
            try:
                response = requests.get(metadata.full_storage_url)
                response.raise_for_status()  # Check that the request was successful
                respons_json = response.json()  # update checkpoints key to use FileInfo
                return respons_json
            except:
                return None

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
