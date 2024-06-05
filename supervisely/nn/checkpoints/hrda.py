from typing import List, Literal

from supervisely.nn.checkpoints.checkpoint import BaseCheckpoint, CheckpointInfo


class HRDACheckpoint(BaseCheckpoint):
    # not enough info to implement

    def __init__(self, team_id: int):
        raise NotImplementedError
        # super().__init__(team_id)
        # self._training_app = "Train HRDA"
        # self._model_dir = "/HRDA"
        # self._weights_dir = None
        # self._task_type = "semantic segmentation"
        # self._weights_ext = ".pth"
        # self._config_file = "config.py"

    def get_session_id(self, session_path: str) -> str:
        raise NotImplementedError

    def get_training_project_name(self, session_path: str) -> str:
        raise NotImplementedError

    def get_task_type(self, session_path: str) -> str:
        raise NotImplementedError

    def get_weights_path(self, session_path: str) -> str:
        raise NotImplementedError

    def get_config_path(self, session_path: str) -> str:
        raise NotImplementedError

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        raise NotImplementedError
