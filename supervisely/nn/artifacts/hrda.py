from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class HRDA(BaseTrainArtifacts):
    # not enough info to implement

    def __init__(self, team_id: int):
        raise NotImplementedError
        # super().__init__(team_id)
        # self._app_name = "Train HRDA"
        # self._artifacts_folder = "/HRDA"
        # self._weights_folder = None
        # self._cv_task = "semantic segmentation"
        # self._weights_ext = ".pth"
        # self._config_file = "config.py"

    def get_task_id(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_project_name(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_cv_task(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_weights_folder(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_config_path(self, artifacts_folder: str) -> str:
        raise NotImplementedError
