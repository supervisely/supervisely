from os.path import join
from re import compile as re_compile

from supervisely.nn.checkpoints.checkpoint import BaseCheckpoint


class YOLOv5Checkpoint(BaseCheckpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train YOLOv5"
        self._model_dir = "/yolov5_train"
        self._weights_dir = "weights"
        self._task_type = "object detection"
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(r"^/yolov5_train/[^/]+/\d+/?$")

    def get_session_id(self, session_path: str) -> str:
        return session_path.split("/")[-1]

    def get_training_project_name(self, session_path: str) -> str:
        return session_path.split("/")[-2]

    def get_task_type(self, session_path: str) -> str:
        return self._task_type

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)

    def get_config_path(self, session_path: str) -> str:
        return None


class YOLOv5v2Checkpoint(YOLOv5Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train YOLOv5 2.0"
        self._model_dir = "/yolov5_2.0_train"
        self._weights_dir = "weights"
        self._task_type = "object detection"
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(r"^/yolov5_2.0_train/[^/]+/\d+/?$")
