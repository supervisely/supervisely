from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class MMClassification(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMClassification"
        self._slug = "supervisely-ecosystem/mmclassification/supervisely/train"
        self._serve_app_name = "Serve MMClassification"
        self._serve_slug = "supervisely-ecosystem/mmclassification/supervisely/serve"
        self._framework_name = "MMClassification"
        self._framework_folder = "/mmclassification"
        self._weights_folder = "checkpoints"
        self._task_type = "classification"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/mmclassification/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["classification"]
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
        return None


class MMPretrain(MMClassification):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMPretrain"
        self._slug = "supervisely-ecosystem/mmpretrain/supervisely/train"
        self._serve_app_name = "Serve MMPretrain"
        self._serve_slug = "supervisely-ecosystem/mmpretrain/supervisely/serve"
        self._framework_name = "MMPretrain"
        self._framework_folder = "/mmclassification-v2"
        self._weights_folder = "checkpoints"
        self._task_type = "classification"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/mmclassification-v2/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["classification"]
        self._require_runtime = False
        self._has_benchmark_evaluation = False
