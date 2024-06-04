from re import compile as re_compile
from time import time
from collections import defaultdict
from os.path import join, basename, dirname
from typing import List, Literal

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo, Checkpoint


class RITMCheckpoint(Checkpoint):
    def __init__(self, team_id: int):
        super().__init__(team_id)

        self._training_app = "Train RITM"
        self._model_dir = "/RITM_training"
        self._weights_dir = "checkpoints"
        self._task_type = None
        self._info_file = "info/ui_state.json"
        self._weights_ext = ".pth"
        self._pattern = re_compile(r"^/RITM_training/\d+_[^/]+/?$")

    def is_valid_session_path(self, path):
        return self._pattern.match(path) is not None

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        def _extract_session_id_and_project_name(path):
            parts = path.split("/")
            if len(parts) < 3:
                raise ValueError(f"Invalid path: {path}")
            session_id, project_name = parts[2].split("_", 1)
            return session_id, project_name

        def _get_task_type_from_info(session_path, info_file):
            info_path = join(session_path, info_file)
            task_type = "undefined"
            for file_info in all_file_infos:
                if file_info.path == info_path:
                    json_data = self._fetch_json_from_url(file_info.full_storage_url)
                    task_type = json_data.get("segmentationType", None)
                    if task_type is not None:
                        task_type = task_type.lower()
                    break
            return task_type

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
                weights_path = join(session_path, self._weights_dir)
                task_type = _get_task_type_from_info(session_path, self._info_file)
                session_id, training_project_name = (
                    _extract_session_id_and_project_name(session_path)
                )
                json_data = self._add_sly_metadata(
                    app_name=self._training_app,
                    session_id=session_id,
                    session_path=session_path,
                    weights_path=weights_path,
                    weights_ext=self._weights_ext,
                    training_project_name=training_project_name,
                    task_type=task_type,
                )
            else:
                for file_info in file_infos:
                    if file_info.path == metadata_path:
                        json_data = self._fetch_json_from_url(
                            file_info.full_storage_url
                        )
                        break
            if json_data is None:
                continue
            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        end_val_time = time()
        print(f"List time: '{end_val_time - start_val_time}' sec")
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
        self._pattern = re_compile(r"^/mmdetection-3/\d+_[^/]+/?$")

    def is_valid_session_path(self, path):
        return self._pattern.match(path) is not None

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        def _extract_session_id(path):
            parts = path.split("/")
            if len(parts) < 3:
                raise ValueError(f"Invalid path: {path}")
            session_id, _ = parts[2].split("_", 1)
            return session_id

        def _get_task_type_and_project_from_config(
            api, team_id, session_path, config_file
        ):
            config_path = join(session_path, config_file)
            api.file.download(team_id, config_path, "model_config.txt")
            task_type = "undefined"
            project_name = None
            with open("model_config.txt", "r") as f:
                lines = f.readlines()
                project_line = lines[-1] if lines else None
                start = project_line.find("'") + 1
                end = project_line.find("'", start)
                project_name = project_line[start:end]
                task_type_line = lines[-3] if lines else None
                start = task_type_line.find("'") + 1
                end = task_type_line.find("'", start)
                task_type = task_type_line[start:end].replace("_", " ")
                f.close()
            silent_remove("model_config.txt")
            return task_type, project_name

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
                weights_path = session_path
                task_type, training_project_name = (
                    _get_task_type_and_project_from_config(
                        self._api, self._team_id, session_path, self._config_file
                    )
                )
                session_id = _extract_session_id(session_path)
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
                for file_info in file_infos:
                    if file_info.path == metadata_path:
                        json_data = self._fetch_json_from_url(
                            file_info.full_storage_url
                        )
                        break
            if json_data is None:
                continue
            checkpoint_info = CheckpointInfo(**json_data)
            checkpoints.append(checkpoint_info)

        end_val_time = time()
        print(f"List time: '{end_val_time - start_val_time}' sec")
        return checkpoints
