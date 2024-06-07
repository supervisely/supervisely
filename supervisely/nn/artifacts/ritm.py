from os.path import join
from re import compile as re_compile

from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


class RITM(BaseTrainArtifacts):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train RITM"
        self._framework_folder = "/RITM_training"
        self._weights_folder = "checkpoints"
        self._task_type = None
        self._info_file = "info/ui_state.json"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/RITM_training/\d+_[^/]+/?$")

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
                json_data = self._fetch_json_from_url(file_info.full_storage_url)
                task_type = json_data.get("segmentationType", "undefined")
                if task_type is not None:
                    task_type = task_type.lower()
                break
        return task_type

    def get_weights_path(self, artifacts_folder: str) -> str:
        return join(artifacts_folder, self._weights_folder)

    def get_config_path(self, artifacts_folder: str) -> str:
        return None
