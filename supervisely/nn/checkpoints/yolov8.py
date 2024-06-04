from re import compile as re_compile
from os.path import basename, join, dirname
from collections import defaultdict
from typing import List, Literal

from time import time
from supervisely._utils import abs_url, is_development
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo, Checkpoint


class YOLOv8Checkpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train YOLOv8"
        self._model_dir = "/yolov8_train"
        self._weights_dir = "weights"
        self._task_type = None
        self._weights_ext = ".pt"
        self._config_file = None
        self._pattern = re_compile(
            r"^/yolov8_train/(object detection|instance segmentation|pose estimation)/[^/]+/\d+/?$"
        )

    def get_model_dir(self):
        return self._model_dir

    def is_valid_session_path(self, path):
        return self._pattern.match(path) is not None

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")

        start_val_time = time()
        all_file_infos = self._api.file.list(
            self._team_id, self._model_dir, return_type="fileinfo"
        )

        folders = defaultdict(set)
        for file_info in all_file_infos:
            session_path = dirname(file_info.path)
            if self.is_valid_session_path(session_path):
                folders[session_path].add(file_info)

        checkpoints = []
        for session_path, file_infos in folders.items():
            file_paths = [file_info.path for file_info in file_infos]
            metadata_path = join(session_path, self._metadata_file_name)
            if metadata_path not in file_paths:
                parts = session_path.split("/")
                weights_path = join(session_path, self._weights_dir)
                training_project_name = parts[-2]
                task_type = parts[2]
                session_id = parts[-1]
                json_data = self._add_sly_metadata(
                    app_name=self._training_app,
                    session_id=session_id,
                    session_path=session_path,
                    weights_path=weights_path,
                    weights_ext=self._weights_ext,
                    training_project_name=training_project_name,
                    task_type=task_type,
                    config_path=None,
                )
            else:
                for file_info in file_infos:
                    if file_info.path == metadata_path:
                        json_data = self._fetch_json_from_url(
                            file_info.full_storage_url
                        )
                        break

            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        end_val_time = time()
        print(f"List time: '{end_val_time - start_val_time}' sec")
        self._sort_checkpoints(checkpoints, sort)
        return checkpoints
