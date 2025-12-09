from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class Detectron2(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train Detectron2"
        self._slug = "supervisely-ecosystem/detectron2/supervisely/train"
        self._serve_app_name = "Serve Detectron2"
        self._serve_slug = (
            "supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve"
        )
        self._framework_name = "Detectron2"
        self._framework_folder = "/detectron2"
        self._weights_folder = "checkpoints"
        self._legacy_weights_folder = "detectron_data"
        self._task_type = "instance segmentation"
        self._weights_ext = ".pth"
        self._config_file = "model_config.yaml"
        self._pattern = re_compile(r"^/detectron2/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["instance segmentation"]
        self._require_runtime = False
        self._has_benchmark_evaluation = False

    def get_task_id(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {artifacts_folder}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_project_name(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {artifacts_folder}")
        _, project_name = parts[2].split("_", 1)
        return project_name

    def get_task_type(self, artifacts_folder: str) -> str:
        return self._task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder, self._config_file)
