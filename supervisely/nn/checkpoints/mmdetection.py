from os.path import join
from typing import List, Literal

from supervisely._utils import abs_url, is_development
from supervisely.io.fs import silent_remove
from supervisely.io.json import load_json_file
from supervisely.nn.checkpoints.checkpoint import Checkpoint, CheckpointInfo


class MMDetectionCheckpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train MMDetection"
        self._model_dir = "/mmdetection"
        self._weights_dir = "checkpoints/data"
        self._task_type = None
        self._weights_ext = ".pth"
        self._config_file = "config.py"

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")
        if not self._api.file.dir_exists(self._team_id, self._model_dir):
            return []

        checkpoints = []
        info_dir_name = "info"
        task_files_infos = self._api.file.list(
            self._team_id, self._model_dir, recursive=False, return_type="fileinfo"
        )
        for task_file_info in task_files_infos:
            json_data = self._fetch_json_from_url(
                f"{task_file_info.path}/{self._metadata_file_name}"
            )
            if json_data is None:
                task_type = None
                path_to_info = join(task_file_info.path, info_dir_name, "ui_state.json")
                if self._api.file.exists(self._team_id, path_to_info):
                    self._api.file.download(self._team_id, path_to_info, "model_config.json")
                    model_config = load_json_file("model_config.json")
                    task_type = model_config["task"].replace("_", " ")
                    silent_remove("model_config.json")
                else:
                    continue

                # config_url = join(path_to_checkpoints, "config.py")
                # if not self._api.file.exists(self._team_id, config_url):
                # continue

                json_data = self._generate_sly_metadata(
                    task_file_info.path,
                    self._weights_dir,
                    self._weights_ext,
                    self._training_app,
                    task_type,
                )
                if json_data is None:
                    continue
            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        if sort == "desc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id, reverse=True)
        elif sort == "asc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id)
        return checkpoints


class MMDetection3Checkpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train MMDetection 3.0"
        self._model_dir = "/mmdetection-3"
        self._weights_dir = None
        self._task_type = None
        self._weights_ext = ".pth"
        self._config_file = "config.py"

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")
        if not self._api.file.dir_exists(self._team_id, self._model_dir):
            return []

        checkpoints = []
        task_files_infos = self._api.file.list(
            self._team_id, self._model_dir, recursive=False, return_type="fileinfo"
        )
        for task_file_info in task_files_infos:
            json_data = self._fetch_json_from_url(
                f"{task_file_info.path}/{self._metadata_file_name}"
            )
            if json_data is None:
                json_data = self._generate_sly_metadata(
                    task_file_info.path,
                    self._weights_dir,
                    self._weights_ext,
                    self._training_app,
                    self._task_type,
                    self._config_file,
                )
                if json_data is None:
                    continue
            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        if sort == "desc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id, reverse=True)
        elif sort == "asc":
            checkpoints = sorted(checkpoints, key=lambda x: x.session_id)
        return checkpoints
