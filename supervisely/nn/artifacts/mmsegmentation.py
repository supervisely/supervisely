from os.path import join
from re import compile as re_compile

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class MMSegmentation(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMSegmentation"
        self._framework_folder = "/mmsegmentation"
        self._weights_folder = "checkpoints/data"
        self._task_type = "instance segmentation"
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmsegmentation/\d+_[^/]+/?$")

    def get_task_id(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[2].split("_")[0]

    def get_project_name(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[2].split("_")[1]

    def get_task_type(self, artifacts_folder: str) -> str:
        return self._task_type

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder, self._config_file)

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)
