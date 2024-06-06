from os.path import join
from re import compile as re_compile

from supervisely.nn.models.base_model import BaseModel


class MMSegmentation(BaseModel):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMSegmentation"
        self._model_dir = "/mmsegmentation"
        self._weights_dir = "checkpoints/data"
        self._task_type = "instance segmentation"
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmsegmentation/\d+_[^/]+/?$")

    def get_session_id(self, session_path: str) -> str:
        return session_path.split("/")[2].split("_")[0]

    def get_training_project_name(self, session_path: str) -> str:
        return session_path.split("/")[2].split("_")[1]

    def get_task_type(self, session_path: str) -> str:
        return self._task_type

    def get_config_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir, self._config_file)

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)
