from abc import abstractmethod
from collections import defaultdict
from json import JSONDecodeError
from os.path import dirname, join
from time import time
from typing import Any, Dict, List, Literal, NamedTuple

import requests

from supervisely import logger
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo
from supervisely.io.fs import silent_remove
from supervisely.io.json import dump_json_file


class CheckpointInfo(NamedTuple):
    """
    CheckpointInfo
    """

    app_name: str
    session_id: int
    session_path: str
    session_link: str
    task_type: str
    training_project_name: str
    checkpoints: List[FileInfo]
    config: str = None


class BaseCheckpoint:
    def __init__(self, team_id: int):
        """
        This is a base class and is not intended to be instantiated directly.
        Subclasses should implement the abstract methods.

        :param team_id: The team ID.
        :type team_id: int
        :raises Exception: If the class is instantiated directly.
        """
        if type(self) is BaseCheckpoint:
            raise Exception(
                "BaseCheckpoint is a base class and should not be instantiated directly"
            )

        self._api: Api = Api.from_env()
        self._team_id: int = team_id
        self._metadata_file_name = "sly_metadata.json"
        self._http_session = requests.Session()

        self._training_app = None
        self._model_dir = None
        self._weights_dir = None
        self._task_type = None
        self._weights_ext = None
        self._config_file = None
        self._pattern = None

    @property
    def team_id(self) -> int:
        """
        Get the team ID.

        :return: The team ID.
        :rtype: int
        """
        return self._team_id

    @property
    def metadata_file_name(self) -> str:
        """
        Metadata file name.

        :return: The metadata file name.
        :rtype: str
        """
        return self._metadata_file_name

    def get_model_dir(self):
        """
        Get the model directory.

        :return: The model directory.
        :rtype: str
        """
        return self._model_dir

    def is_valid_session_path(self, path):
        """
        Check if the provided path is valid and follows specified session path pattern.

        :param path: The session path.
        :type path: str
        :return: True if the session path is valid, False otherwise.
        :rtype: bool
        """
        return self._pattern.match(path) is not None

    @abstractmethod
    def get_session_id(self, session_path: str) -> str:
        """
        Get the session ID.

        :param session_path: The session path.
        :type session_path: str
        :return: The session ID.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_training_project_name(self, session_path: str) -> str:
        """
        Get the training project name.

        :param session_path: The session path.
        :type session_path: str
        :return: The training project name.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_task_type(self, session_path: str) -> str:
        """
        Get the task type.

        :param session_path: The session path.
        :type session_path: str
        :return: The task type.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_weights_path(self, session_path: str) -> str:
        """
        Get path to weights directory.

        :param session_path: The session path.
        :type session_path: str
        :return: The weights directory path.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_config_path(self, session_path: str) -> str:
        """
        Get path to config file.

        :param session_path: The session path.
        :type session_path: str
        :return: The config file path.
        :rtype: str
        """
        pass

    def sort_checkpoints(
        self, checkpoints: List[CheckpointInfo], sort: Literal["desc", "asc"] = "desc"
    ) -> List[CheckpointInfo]:
        """
        Sort the checkpoints.

        :param checkpoints: The list of checkpoints.
        :type checkpoints: List[FileInfo]
        :param sort: The sort order, either "desc" or "asc". Default is "desc", which means newer checkpoints will be first.
        :type sort: Literal["desc", "asc"]
        :return: The sorted list of checkpoints.
        :rtype: List[CheckpointInfo]
        """
        checkpoints_with_ids = [(c.session_id, c) for c in checkpoints]
        if sort == "desc":
            checkpoints_with_ids.sort(reverse=True)
        elif sort == "asc":
            checkpoints_with_ids.sort()
        checkpoints = [c for _, c in checkpoints_with_ids]
        return checkpoints

    def remove_sly_metadata(self, session_path: str) -> None:
        """
        Remove the metadata file from the session folder.

        :param session_path: The session path.
        :type session_path: str
        """
        metadata_path = join(session_path, self._metadata_file_name)
        self._api.file.remove(self._team_id, metadata_path)
        logger.info(f"File '{metadata_path}' was removed")

    def remove_sly_metadatas(self) -> None:
        """
        Remove the metadata files from the session folders.

        :param session_paths: The session paths.
        :type session_paths: List[str]
        """
        count = 0
        file_paths = [file_info.path for file_info in self._get_file_infos()]
        for file_path in file_paths:
            if file_path.endswith("sly_metadata.json"):
                self._api.file.remove(self._team_id, file_path)
                logger.info(f"File '{file_path}' was removed")
                count += 1
        logger.info(f"Total files removed: '{count}'")

    def generate_sly_metadata(
        self,
        app_name: str,
        session_id: str,
        session_path: str,
        weights_path: str,
        weights_ext: str,
        training_project_name: str,
        task_type: str = None,
        config_path: str = None,
    ):
        """
        Generate the metadata for the given parameters.

        :param app_name: Name of the training application.
        :type app_name: str
        :param session_id: The session ID.
        :type session_id: str
        :param session_path: Path to session folder.
        :type session_path: str
        :param weights_path: Path to weights location.
        :type weights_path: str
        :param weights_ext: The weights extension.
        :type weights_ext: str
        :param training_project_name: Name of project used for training.
        :type training_project_name: str
        :param task_type: The task type. Default is None.
        :type task_type: str, optional
        :param config_path: Path to config file. Default is None.
        :type config_path: str, optional
        :return: The generated metadata.
        :rtype: dict
        """

        def _get_checkpoint_file_infos(weights_path) -> List[FileInfo]:
            return [
                file
                for file in self._api.file.list(
                    self._team_id,
                    weights_path,
                    recursive=False,
                    return_type="fileinfo",
                )
                if file.name.endswith(weights_ext)
            ]

        def _upload_metadata(json_data: dict) -> None:
            json_data_path = self._metadata_file_name
            dump_json_file(json_data, json_data_path)
            self._api.file.upload(
                self._team_id,
                json_data_path,
                f"{session_path}/{self._metadata_file_name}",
            )
            silent_remove(json_data_path)

        checkpoint_file_infos = _get_checkpoint_file_infos(weights_path)
        if len(checkpoint_file_infos) == 0:
            logger.info(f"No checkpoints found in '{session_path}'")
            return None

        logger.info(f"Generating '{self._metadata_file_name}' for '{session_path}'")
        if is_development():
            session_link = abs_url(f"/apps/sessions/{session_id}")
        else:
            session_link = f"/apps/sessions/{session_id}"

        checkpoint_json = {
            "app_name": app_name,
            "session_id": session_id,
            "session_path": session_path,
            "session_link": session_link,
            "task_type": task_type,
            "training_project_name": training_project_name,
            "checkpoints": checkpoint_file_infos,
        }
        if config_path is not None:
            checkpoint_json["config"] = config_path
        _upload_metadata(checkpoint_json)
        return checkpoint_json

    def _fetch_json_from_url(self, metadata_url: str):
        try:
            response = self._http_session.get(metadata_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch model metadata from '{metadata_url}': {e}")
            return None

        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.debug(f"Failed to decode JSON from '{metadata_url}': {e}")
            return None

        checkpoints = response_json.get("checkpoints", [])
        file_infos = [FileInfo(*checkpoint) for checkpoint in checkpoints]
        response_json["checkpoints"] = file_infos

        return response_json

    def _get_checkpoint_json(
        self,
        session_path: str,
        metadata_path: str,
        file_infos: List[FileInfo],
        file_paths: List[str],
    ) -> Dict[str, Any]:
        json_data = None
        if metadata_path not in file_paths:
            weights_path = self.get_weights_path(session_path)
            task_type = self.get_task_type(session_path)
            session_id = self.get_session_id(session_path)
            training_project_name = self.get_training_project_name(session_path)
            config_path = self.get_config_path(session_path)
            json_data = self.generate_sly_metadata(
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
                    json_data = self._fetch_json_from_url(file_info.full_storage_url)
                    break
            end_find_time = time()
            logger.debug(
                f"Fetch metadata for {metadata_path}: '{format(end_find_time - start_find_time, '.6f')}' sec"
            )
        return json_data

    def _validate_sort(self, sort: Literal["desc", "asc"]):
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")

    def _get_file_infos(self):
        return self._api.file.list(self._team_id, self._model_dir, return_type="fileinfo")

    def _group_files_by_folder(self, file_infos: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        folders = defaultdict(list)
        for file_info in file_infos:
            session_path = dirname(file_info.path)
            if self.is_valid_session_path(session_path):
                folders[session_path].append(file_info)
        return folders

    def _create_checkpoints(self, folders):
        checkpoints = []
        for session_path, file_infos in folders.items():
            metadata_path = join(session_path, self._metadata_file_name)
            file_paths = [file_info.path for file_info in file_infos]
            checkpoint_json = self._get_checkpoint_json(
                session_path, metadata_path, file_infos, file_paths
            )
            if checkpoint_json is None:
                continue
            checkpoint_info = CheckpointInfo(**checkpoint_json)
            checkpoints.append(checkpoint_info)
        return checkpoints

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        """
        Get the list of checkpoints.

        :param sort: The sort order, either "desc" or "asc". Default is "desc".
        :type sort: Literal["desc", "asc"]
        :return: The list of checkpoints.
        :rtype: List[CheckpointInfo]
        """
        self._validate_sort(sort)
        start_time = time()
        parsed_infos = self._get_file_infos()
        folders = self._group_files_by_folder(parsed_infos)
        checkpoints = self._create_checkpoints(folders)
        end_time = time()
        self.sort_checkpoints(checkpoints, sort)
        logger.debug(f"Listing time: '{format(end_time - start_time, '.6f')}' sec")
        return checkpoints
