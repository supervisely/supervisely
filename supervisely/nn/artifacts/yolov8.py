from os.path import join
from re import compile as re_compile

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class YOLOv8(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train YOLOv8"
        self._framework_folder = "/yolov8_train"
        self._weights_folder = "weights"
        self._task_type = None
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(
            r"^/yolov8_train/(object detection|instance segmentation|pose estimation)/[^/]+/\d+/?$"
        )

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
