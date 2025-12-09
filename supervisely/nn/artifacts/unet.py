from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class UNet(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train UNet"
        self._slug = "supervisely-ecosystem/unet/supervisely/train"
        self._serve_app_name = "Serve UNet"
        self._serve_slug = "supervisely-ecosystem/unet/supervisely/serve"
        self._framework_name = "UNet"
        self._framework_folder = "/unet"
        self._weights_folder = "checkpoints"
        self._task_type = "semantic segmentation"
        self._weights_ext = ".pth"
        self._config_file = "train_args.json"
        self._pattern = re_compile(r"^/unet/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["semantic segmentation"]
        self._require_runtime = False
        self._has_benchmark_evaluation = True

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
