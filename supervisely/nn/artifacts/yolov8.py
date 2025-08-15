from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class YOLOv8(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train YOLOv8 | v9 | v10 | v11"
        self._slug = "supervisely-ecosystem/yolov8/train"
        self._serve_app_name = "Serve YOLOv8 | v9 | v10 | v11"
        self._serve_slug = "supervisely-ecosystem/yolov8/serve"
        self._framework_name = "YOLOv8"
        self._framework_folder = "/yolov8_train"
        self._weights_folder = "weights"
        self._task_type = None
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(
            r"^/yolov8_train/(object detection|instance segmentation|pose estimation)/[^/]+/\d+/?$"
        )
        self._available_task_types: List[str] = [
            "object detection",
            "instance segmentation",
            "pose estimation",
        ]
        self._require_runtime = True
        self._has_benchmark_evaluation = True

    def get_task_id(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        return parts[-1]

    def get_project_name(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        return parts[-2]

    def get_task_type(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        return parts[2]

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return None
