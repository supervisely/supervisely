from supervisely.nn.models.base_model import BaseModel


class HRDA(BaseModel):
    # not enough info to implement

    def __init__(self, team_id: int):
        raise NotImplementedError
        # super().__init__(team_id)
        # self._app_name = "Train HRDA"
        # self._framework_dir = "/HRDA"
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
