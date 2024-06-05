from os.path import join
from re import compile as re_compile

from supervisely.nn.checkpoints.checkpoint import BaseCheckpoint


class MMCClassificationCheckpoint(BaseCheckpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMClassification"
        self._model_dir = "/mmclassification"
        self._weights_dir = "checkpoints"
        self._task_type = "classification"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/mmclassification/\d+_[^/]+/?$")

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
        return None
