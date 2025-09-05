from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class RTDETR(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train RT-DETR"
        self._slug = "supervisely-ecosystem/rt-detr/supervisely_integration/train"
        self._serve_app_name = "Serve RT-DETR"
        self._serve_slug = "supervisely-ecosystem/rt-detr/supervisely_integration/serve"
        self._framework_name = "RT-DETR"
        self._framework_folder = "/RT-DETR"
        self._weights_folder = "weights"
        self._task_type = "object detection"
        self._weights_ext = ".pth"
        self._config_file = "config.yml"
        self._pattern = re_compile(r"^/RT-DETR/[^/]+/\d+/?$")
        self._available_task_types: List[str] = ["object detection"]
        self._require_runtime = False
        self._has_benchmark_evaluation = True

    def get_task_id(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[-1]

    def get_project_name(self, artifacts_folder: str) -> str:
        return artifacts_folder.split("/")[-2]

    def get_task_type(self, artifacts_folder: str) -> str:
        return self._task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._config_file)
