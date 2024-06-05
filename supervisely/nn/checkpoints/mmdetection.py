from os.path import join
from re import compile as re_compile

from supervisely.io.fs import silent_remove
from supervisely.nn.checkpoints.checkpoint import BaseCheckpoint


class MMDetectionCheckpoint(BaseCheckpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection"
        self._model_dir = "/mmdetection"
        self._weights_dir = "checkpoints/data"
        self._task_type = None
        self._weights_ext = ".pth"
        self._info_file = "info/ui_state.json"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection/\d+_[^/]+/?$")

    def get_session_id(self, session_path: str) -> str:
        parts = session_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {session_path}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_training_project_name(self, session_path: str) -> str:
        parts = session_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {session_path}")
        _, project_name = parts[2].split("_", 1)
        return project_name

    def get_task_type(self, session_path: str) -> str:
        info_path = join(session_path, self._info_file)
        task_type = "undefined"
        for file_info in self._get_file_infos():
            if file_info.path == info_path:
                json_data = self._fetch_json_from_url(file_info.full_storage_url)
                task_type = json_data.get("task", "undefined")
                break
        return task_type

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)

    def get_config_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir, self._config_file)


class MMDetection3Checkpoint(BaseCheckpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train MMDetection 3.0"
        self._model_dir = "/mmdetection-3"
        self._weights_dir = None
        self._task_type = None
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmdetection-3/\d+_[^/]+/?$")

    def get_session_id(self, session_path: str) -> str:
        parts = session_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid path: {session_path}")
        session_id, _ = parts[2].split("_", 1)
        return session_id

    def get_training_project_name(self, session_path: str) -> str:
        config_path = join(session_path, self._config_file)
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

    def get_task_type(self, session_path: str) -> str:
        config_path = join(session_path, self._config_file)
        self._api.file.download(self._team_id, config_path, "model_config.txt")
        task_type = "undefined"
        with open("model_config.txt", "r") as f:
            lines = f.readlines()
            task_type_line = lines[-3] if lines else None
            start = task_type_line.find("'") + 1
            end = task_type_line.find("'", start)
            task_type = task_type_line[start:end].replace("_", " ")
            f.close()
        silent_remove("model_config.txt")
        return task_type

    def get_weights_path(self, session_path: str) -> str:
        return session_path

    def get_config_path(self, session_path: str) -> str:
        return join(session_path, self._config_file)
