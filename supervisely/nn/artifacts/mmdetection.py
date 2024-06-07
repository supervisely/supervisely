from os.path import join
from re import compile as re_compile

from supervisely.io.fs import silent_remove
from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class MMDetection(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection"
        self._framework_folder = "/mmdetection"
        self._weights_folder = "checkpoints/data"
        self._cv_task = None
        self._weights_ext = ".pth"
        self._info_file = "info/ui_state.json"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection/\d+_[^/]+/?$")

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

    def get_cv_task(self, artifacts_folder: str) -> str:
        info_path = join(artifacts_folder, self._info_file)
        cv_task = "undefined"
        for file_info in self._get_file_infos():
            if file_info.path == info_path:
                json_data = self._fetch_json_from_url(file_info.full_storage_url)
                cv_task = json_data.get("task", "undefined")
                break
        return cv_task

    def get_weights_folder(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder, self._config_file)


class MMDetection3(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection 3.0"
        self._framework_folder = "/mmdetection-3"
        self._weights_folder = None
        self._cv_task = None
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection-3/\d+_[^/]+/?$")

    def get_task_id(self, artifacts_folder: str) -> str:
        parts = artifacts_folder.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {artifacts_folder}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_project_name(self, artifacts_folder: str) -> str:
        config_path = join(artifacts_folder, self._config_file)
        self._api.file.download(self._team_id, config_path, "model_config.txt")
        project_name = None
        with open("model_config.txt", "r") as f:
            lines = f.readlines()
            project_line = lines[-1] if lines else None
            start = project_line.find("'") + 1
            end = project_line.find("'", start)
            project_name = project_line[start:end]
            f.close()
        silent_remove("model_config.txt")
        return project_name

    def get_cv_task(self, artifacts_folder: str) -> str:
        config_path = join(artifacts_folder, self._config_file)
        self._api.file.download(self._team_id, config_path, "model_config.txt")
        cv_task = "undefined"
        with open("model_config.txt", "r") as f:
            lines = f.readlines()
            cv_task_line = lines[-3] if lines else None
            start = cv_task_line.find("'") + 1
            end = cv_task_line.find("'", start)
            cv_task = cv_task_line[start:end].replace("_", " ")
            f.close()
        silent_remove("model_config.txt")
        return cv_task

    def get_weights_folder(self, artifacts_folder: str) -> str:
        return artifacts_folder

    def get_config_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._config_file)
