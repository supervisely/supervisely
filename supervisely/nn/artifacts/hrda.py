from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class HRDA(BaseTrainArtifacts):
    # not enough info to implement

    def __init__(self, team_id: int):
        raise NotImplementedError
        # super().__init__(team_id)
        # self._app_name = "Train HRDA"
        # self._serve_app_name = None
        # self._slug = None
        # self._serve_slug = None
        # self._framework_folder = "/HRDA"
        # self._weights_folder = None
        # self._task_type = "semantic segmentation"
        # self._weights_ext = ".pth"
        # self._config_file = "config.py"
        # self._available_task_types: List[str] = ["semantic segmentation"]
        # self._require_runtime = False
        # self._has_benchmark_evaluation = False

    def get_task_id(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_project_name(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_task_type(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_weights_path(self, artifacts_folder: str) -> str:
        raise NotImplementedError

    def get_config_path(self, artifacts_folder: str) -> str:
        raise NotImplementedError
