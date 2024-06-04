from os.path import join
from typing import List, Literal

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo, Checkpoint


class MMCClassificationCheckpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)
        
        self._training_app = "Train MMClassification"
        self._model_dir = "/mmclassification"
        self._weights_dir = "checkpoints"
        self._task_type = "classification"
        self._weights_ext = ".pth"
        # self._config_file = "config.py"
        
        
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
