from re import compile as re_compile
from time import time
from collections import defaultdict
from os.path import join, basename, dirname
from typing import List, Literal
from supervisely import logger
from supervisely.api.api import Api
from supervisely._utils import abs_url, is_development
from supervisely.nn.checkpoints.checkpoint import Checkpoint, CheckpointInfo


class MMSegmentationCheckpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train MMSegmentation"
        self._model_dir = "/mmsegmentation"
        self._weights_dir = "checkpoints/data"
        self._task_type = "instance segmentation"
        self._weights_ext = ".pth"
        self._config_file = "config.py"
        self._pattern = re_compile(r"^/mmsegmentation/\d+_[^/]+/?$")

    def is_valid_session_path(self, path):
        return self._pattern.match(path) is not None

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        def _extract_session_id_and_project_name(path):
            parts = path.split("/")
            if len(parts) < 3:
                raise ValueError(f"Invalid path: {path}")
            session_id, project_name = parts[2].split("_", 1)
            return session_id, project_name

        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")

        start_val_time = time()
        file_infos = self._api.file.list(
            self._team_id, self._model_dir, return_type="fileinfo"
        )

        folders = defaultdict(set)
        for file_info in file_infos:
            session_path = dirname(file_info.path)
            if self.is_valid_session_path(session_path):
                folders[session_path].add(file_info)

        checkpoints = []
        for session_path, file_infos in folders.items():
            file_paths = [file_info.path for file_info in file_infos]
            metadata_path = join(session_path, self._metadata_file_name)
            if metadata_path not in file_paths:
                weights_path = join(session_path, self._weights_dir)
                task_type = self._task_type
                session_id, training_project_name = (
                    _extract_session_id_and_project_name(session_path)
                )
                config_path = join(weights_path, self._config_file)

                json_data = self._add_sly_metadata(
                    app_name=self._training_app,
                    session_id=session_id,
                    session_path=session_path,
                    weights_path=weights_path,
                    weights_ext=self._weights_ext,
                    training_project_name=training_project_name,
                    task_type=task_type,
                    config_path=config_path,
                )
            else:
                start_find_time = time()
                for file_info in file_infos:
                    if file_info.path == metadata_path:
                        json_data = self._fetch_json_from_url(
                            file_info.full_storage_url
                        )
                        break
                end_find_time = time()
                print(f"Fetch metadata for {metadata_path}: '{format(end_find_time - start_find_time, '.6f')}' sec")
                    
            if json_data is None:
                continue
            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        end_val_time = time()
        print(f"List time: '{format(end_val_time - start_val_time, '.6f')}' sec")
        
        self._sort_checkpoints(checkpoints, sort)
        return checkpoints
