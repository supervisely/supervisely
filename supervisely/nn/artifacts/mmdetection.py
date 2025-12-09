import random
import string
from os.path import join
from re import compile as re_compile
from typing import List

from supervisely.io.fs import silent_remove
from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class MMDetection(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection"
        self._slug = "supervisely-ecosystem/mmdetection/train"
        self._serve_app_name = "Serve MMDetection"
        self._serve_slug = "supervisely-ecosystem/mmdetection/serve"
        self._framework_name = "MMDetection"
        self._framework_folder = "/mmdetection"
        self._weights_folder = "checkpoints/data"
        self._task_type = None
        self._weights_ext = ".pth"
        self._info_file = "info/ui_state.json"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["object detection", "instance segmentation"]
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
        info_path = join(artifacts_folder, self._info_file)
        task_type = "undefined"
        for file_info in self._get_file_infos():
            if file_info.path == info_path:
                json_data = self._fetch_json_from_path(file_info.path)
                task_type = json_data.get("task", "undefined")
                break
        return task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder, self._config_file)


class MMDetection3(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection 3.0"
        self._slug = "supervisely-ecosystem/train-mmdetection-v3"
        self._serve_app_name = "Serve MMDetection 3.0"
        self._serve_slug = "supervisely-ecosystem/serve-mmdetection-v3"
        self._framework_name = "MMDetection 3.0"
        self._framework_folder = "/mmdetection-3"
        self._weights_folder = None
        self._task_type = None
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection-3/\d+_[^/]+/?$")
        self._available_task_types: List[str] = ["object detection", "instance segmentation"]
        self._require_runtime = False
        self._has_benchmark_evaluation = True

    def get_task_id(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {artifacts_folder}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_project_name(self, artifacts_folder: str) -> str:
        config_path = join(artifacts_folder, self._config_file)
        config_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".txt"
        self._api.file.download(self._team_id, config_path, config_name)
        project_name = None
        with open(config_name, "r") as f:
            lines = f.readlines()
            project_line = lines[-1] if lines else None
            if project_line is None:
                f.close()
                silent_remove(config_name)
                return project_name
            start = project_line.find("'") + 1
            end = project_line.find("'", start)
            project_name = project_line[start:end]
            f.close()
        silent_remove(config_name)
        return project_name

    def get_task_type(self, artifacts_folder: str) -> str:
        config_path = join(artifacts_folder, self._config_file)
        config_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".txt"
        self._api.file.download(self._team_id, config_path, config_name)
        task_type = "undefined"
        with open(config_name, "r") as f:
            lines = f.readlines()
            task_type_line = lines[-3] if lines else None
            if task_type_line is None:
                f.close()
                silent_remove(config_name)
                return task_type
            start = task_type_line.find("'") + 1
            end = task_type_line.find("'", start)
            task_type = task_type_line[start:end].replace("_", " ")
            f.close()
        silent_remove(config_name)
        return task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return artifacts_folder

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._config_file)
