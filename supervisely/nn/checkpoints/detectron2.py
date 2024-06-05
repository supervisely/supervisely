from os.path import join
from re import compile as re_compile

from supervisely.nn.checkpoints.checkpoint import BaseCheckpoint


class Detectron2Checkpoint(BaseCheckpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train Detectron2"
        self._model_dir = "/detectron2"
        self._weights_dir = "detectron_data"
        self._task_type = "instance segmentation"
        self._weights_ext = ".pth"
        self._config_file = "model_config.yaml"
        self._pattern = re_compile(r"^/detectron2/\d+_[^/]+/?$")

    def get_session_id(self, session_path: str) -> str:
        parts = session_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {session_path}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_training_project_name(self, session_path: str) -> str:
        parts = session_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {session_path}")
        _, project_name = parts[2].split("_", 1)
        return project_name

    def get_task_type(self, session_path: str) -> str:
        return self._task_type

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)

    def get_config_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir, self._config_file)
