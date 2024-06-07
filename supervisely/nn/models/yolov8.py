from os.path import join
from re import compile as re_compile

from supervisely.nn.models.base_model import BaseModel


class YOLOv8(BaseModel):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train YOLOv8"
        self._framework_dir = "/yolov8_train"
        self._weights_dir = "weights"
        self._task_type = None
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(
            r"^/yolov8_train/(object detection|instance segmentation|pose estimation)/[^/]+/\d+/?$"
        )

    def get_session_id(self, session_path: str) -> str:
        parts = session_path.split("/")
        return parts[-1]

    def get_training_project_name(self, session_path: str) -> str:
        parts = session_path.split("/")
        return parts[-2]

    def get_task_type(self, session_path: str) -> str:
        parts = session_path.split("/")
        return parts[2]

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)

    def get_config_path(self, session_path: str) -> str:
        return None
