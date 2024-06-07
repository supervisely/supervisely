from os.path import join
from re import compile as re_compile

from supervisely.nn.models.base_model import BaseModel


class RITM(BaseModel):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._app_name = "Train RITM"
        self._framework_dir = "/RITM_training"
        self._weights_dir = "checkpoints"
        self._task_type = None
        self._info_file = "info/ui_state.json"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/RITM_training/\d+_[^/]+/?$")

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
                task_type = json_data.get("segmentationType", "undefined")
                if task_type is not None:
                    task_type = task_type.lower()
                break
        return task_type

    def get_weights_path(self, session_path: str) -> str:
        return join(session_path, self._weights_dir)

    def get_config_path(self, session_path: str) -> str:
        return None
