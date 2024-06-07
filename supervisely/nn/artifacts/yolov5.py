from os.path import join
from re import compile as re_compile

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class YOLOv5(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train YOLOv5"
        self._framework_folder = "/yolov5_train"
        self._weights_folder = "weights"
        self._task_type = "object detection"
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(r"^/yolov5_train/[^/]+/\d+/?$")

    def get_task_id(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[-1]

    def get_project_name(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[-2]

    def get_task_type(self, artifacts_folder: str) -> str:
        return self._task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return None


class YOLOv5v2(YOLOv5):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train YOLOv5 2.0"
        self._framework_folder = "/yolov5_2.0_train"
        self._weights_folder = "weights"
        self._task_type = "object detection"
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(r"^/yolov5_2.0_train/[^/]+/\d+/?$")
