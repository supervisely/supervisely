from __future__ import annotations
import random
import string
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
from json import JSONDecodeError
from os.path import dirname, join
from time import time
from typing import Any, Dict, List, Literal, NamedTuple, Union
import requests

from supervisely import logger
from supervisely._utils import abs_url, is_development
from supervisely.api.file_api import FileInfo
from supervisely.io.fs import get_file_name_with_ext, silent_remove
from supervisely.io.json import dump_json_file
from supervisely.api.api import Api, ApiField
from supervisely.nn.experiments import ExperimentInfo


class TrainInfo(NamedTuple):
    """
    TrainInfo
    """

    app_name: str
    task_id: int
    artifacts_folder: str
    session_link: str
    task_type: str
    project_name: str
    checkpoints: List[FileInfo]
    config_path: str = None


class BaseTrainArtifacts:
    def __init__(self, team_id: int):
        """
        This is a base class and is not intended to be instantiated directly.
        Subclasses should implement the abstract methods.

        :param team_id: The team ID.
        :type team_id: int
        :raises Exception: If the class is instantiated directly.
        """
        if type(self) is BaseTrainArtifacts:
            raise Exception(
                "BaseTrainArtifacts is a base class and should not be instantiated directly"
            )

        self._api: Api = Api.from_env()
        self._team_id: int = team_id
        self._metadata_file_name: str = "train_info.json"

        self._app_name: str = None
        self._slug = None
        self._serve_app_name = None
        self._serve_slug = None
        self._framework_name: str = None
        self._framework_folder: str = None
        self._weights_folder: str = None
        self._task_type: str = None
        self._weights_ext: str = None
        self._config_file: str = None
        self._pattern: str = None
        self._available_task_types: List[str] = []
        self._require_runtime = False
        self._has_benchmark_evaluation = False

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

    @property
    def app_name(self):
        """
        Train application name.

        :return: The train application name.
        :rtype: str
        """
        return self._app_name

    @property
    def slug(self):
        """
        Train app slug.

        :return: Train app slug.
        :rtype: str
        """
        return self._slug

    @property
    def serve_app_name(self):
        """
        Serve application name.

        :return: The serve application name.
        :rtype: str
        """
        return self._serve_app_name

    @property
    def serve_slug(self):
        """
        Serve app slug.

        :return: Serve app slug.
        :rtype: str
        """
        return self._serve_slug

    @property
    def framework_name(self):
        """
        Framework name.

        :return: The framework name.
        :rtype: str
        """
        return self._framework_name

    @property
    def framework_folder(self):
        """
        Path to framework folder in Supervisely Team Files.

        :return: The framework folder path.
        :rtype: str
        """
        return self._framework_folder

    @property
    def weights_folder(self):
        """
        Weights folder path relative to artifacts folder.

        :return: The weights folder path.
        :rtype: str
        """
        return self._weights_folder

    @property
    def task_type(self):
        """
        Framework computer vision task. None if can be multiple tasks.

        :return: The cv task.
        :rtype: Union[str, None]
        """
        return self._task_type

    @property
    def weights_ext(self):
        """
        Checkpoint weights extension.

        :return: The weights extension.
        :rtype: str
        """
        return self._weights_ext

    @property
    def config_file(self):
        """
        Name of the config file with extension.

        :return: The config file name.
        :rtype: str
        """
        return self._config_file

    @property
    def pattern(self):
        """
        Framework artifacts folder path pattern.

        :return: The artifacts folder path pattern.
        :rtype: re.Pattern
        """
        return self._pattern

    @property
    def require_runtime(self):
        """
        Whether providing runtime is required for the framework.

        :return: True if runtime is required, False otherwise.
        :rtype: bool
        """
        return self._require_runtime

    @property
    def has_benchmark_evaluation(self):
        """
        Whether the framework has integrated benchmark evaluation.
        """
        return self._has_benchmark_evaluation

    def is_valid_artifacts_path(self, path):
        """
        Check if the provided path is valid and follows specified session path pattern.

        :param path: Path to artifacts folder.
        :type path: str
        :return: True if the path to artifacts folder is valid, False otherwise.
        :rtype: bool
        """
        return self._pattern.match(path) is not None

    @abstractmethod
    def get_task_id(self, artifacts_folder: str) -> str:
        """
        Get the task ID of training session.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        :return: The task ID.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_project_name(self, artifacts_folder: str) -> str:
        """
        Get the training project name.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        :return: The training project name.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_task_type(self, artifacts_folder: str) -> str:
        """
        Get the cv task.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        :return: The cv task.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_weights_path(self, artifacts_folder: str) -> str:
        """
        Get path to weights folder.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        :return: The weights folder path.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_config_path(self, artifacts_folder: str) -> str:
        """
        Get path to config file.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        :return: The config file path.
        :rtype: str
        """
        pass

    def sort_train_infos(
        self, train_infos: List[TrainInfo], sort: Literal["desc", "asc"] = "desc"
    ) -> List[TrainInfo]:
        """
        Sort artifacts folder by task id.

        :param train_infos: The list of training infos.
        :type train_infos: List[TrainInfo]
        :param sort: The sort order, either "desc" or "asc". Default is "desc", which means newer checkpoints will be first.
        :type sort: Literal["desc", "asc"]
        :return: The sorted list of checkpoints.
        :rtype: List[TrainInfo]
        """
        if sort == "desc":
            return sorted(train_infos, key=lambda x: int(x.task_id), reverse=True)
        elif sort == "asc":
            return sorted(train_infos, key=lambda x: int(x.task_id))

    def remove_metadata(self, artifacts_folder: str) -> None:
        """
        Remove the metadata file from the session folder.

        :param artifacts_folder: Path to artifacts folder.
        :type artifacts_folder: str
        """
        metadata_path = join(artifacts_folder, self._metadata_file_name)
        self._api.file.remove(self._team_id, metadata_path)
        logger.info(f"File '{metadata_path}' was removed")

    def remove_all_metadatas(self) -> None:
        """
        Remove the metadata files from the session folders.

        :param artifacts_folders: Path to artifacts folders.
        :type artifacts_folders: List[str]
        """
        count = 0
        file_paths = [file_info.path for file_info in self._get_file_infos()]
        for file_path in file_paths:
            if file_path.endswith(self._metadata_file_name):
                self._api.file.remove(self._team_id, file_path)
                logger.info(f"File '{file_path}' was removed")
                count += 1
        logger.info(f"Total files removed: '{count}'")

    def generate_metadata(
        self,
        app_name: str,
        task_id: str,
        artifacts_folder: str,
        weights_folder: str,
        weights_ext: str,
        project_name: str,
        task_type: str = None,
        config_path: str = None,
    ):
        """
        Generate the metadata for the given parameters.

        :param app_name: Name of the training application.
        :type app_name: str
        :param task_id: The session ID.
        :type task_id: str
        :param artifacts_folder: Path to session folder.
        :type artifacts_folder: str
        :param weights_folder: Path to weights location.
        :type weights_folder: str
        :param weights_ext: The weights extension.
        :type weights_ext: str
        :param project_name: Name of project used for training.
        :type project_name: str
        :param task_type: CV Task. Default is None.
        :type task_type: str, optional
        :param config_path: Path to config file. Default is None.
        :type config_path: str, optional
        :return: The generated metadata.
        :rtype: dict
        """

        def _get_checkpoint_file_infos(weights_folder) -> List[FileInfo]:
            return [
                file
                for file in self._api.file.list(
                    self._team_id,
                    weights_folder,
                    recursive=False,
                    return_type="fileinfo",
                )
                if file.name.endswith(weights_ext)
            ]

        def _upload_metadata(json_data: dict) -> None:
            random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
            json_data_path = f"{random_string}.txt"
            dump_json_file(json_data, json_data_path)
            self._api.file.upload(
                self._team_id,
                json_data_path,
                f"{artifacts_folder}/{self._metadata_file_name}",
            )
            silent_remove(json_data_path)

        checkpoint_file_infos = _get_checkpoint_file_infos(weights_folder)
        if hasattr(self, "_legacy_weights_folder"):
            if self._legacy_weights_folder is not None:
                legacy_checkpoint_file_infos = _get_checkpoint_file_infos(
                    self._legacy_weights_folder
                )
                if len(legacy_checkpoint_file_infos) > 0:
                    checkpoint_file_infos.extend(legacy_checkpoint_file_infos)

        if len(checkpoint_file_infos) == 0:
            logger.info(f"No checkpoints found in '{weights_folder}'")
            return None

        logger.info(f"Generating '{self._metadata_file_name}' for '{artifacts_folder}'")
        if is_development():
            session_link = abs_url(f"/apps/sessions/{task_id}")
        else:
            session_link = f"/apps/sessions/{task_id}"

        train_json = {
            "app_name": app_name,
            "task_id": task_id,
            "artifacts_folder": artifacts_folder,
            "session_link": session_link,
            "task_type": task_type,
            "project_name": project_name,
            "checkpoints": checkpoint_file_infos,
        }
        if config_path is not None:
            train_json["config_path"] = config_path
        is_valid = self._validate_train_json(train_json)
        if is_valid:
            _upload_metadata(train_json)
            logger.info(f"Metadata for '{artifacts_folder}' was generated")
        else:
            logger.debug(f"Invalid metadata for '{artifacts_folder}'")
            train_json = None
        return train_json

    def _fetch_json_from_path(self, remote_path: str):
        try:
            response = self._api.post(
                "file-storage.download",
                {ApiField.TEAM_ID: self.team_id, ApiField.PATH: remote_path},
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch train metadata from '{remote_path}': {e}")
            return None

        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.debug(f"Failed to decode JSON from '{remote_path}': {e}")
            return None

        checkpoints = response_json.get("checkpoints", [])
        file_infos = [FileInfo(*checkpoint) for checkpoint in checkpoints]
        response_json["checkpoints"] = file_infos

        return response_json

    def _get_train_json(
        self,
        artifacts_folder: str,
        metadata_path: str,
        file_infos: List[FileInfo],
        file_paths: List[str],
    ) -> Dict[str, Any]:
        json_data = None
        if metadata_path not in file_paths:
            weights_folder = self.get_weights_path(artifacts_folder)
            task_type = self.get_task_type(artifacts_folder)
            task_id = self.get_task_id(artifacts_folder)
            project_name = self.get_project_name(artifacts_folder)
            config_path = self.get_config_path(artifacts_folder)
            json_data = self.generate_metadata(
                app_name=self._app_name,
                task_id=task_id,
                artifacts_folder=artifacts_folder,
                weights_folder=weights_folder,
                weights_ext=self._weights_ext,
                project_name=project_name,
                task_type=task_type,
                config_path=config_path,
            )
        else:
            start_find_time = time()
            for file_info in file_infos:
                if file_info.path == metadata_path:
                    json_data = self._fetch_json_from_path(file_info.path)
                    break
            end_find_time = time()
            logger.debug(
                f"Fetch metadata for {metadata_path}: '{format(end_find_time - start_find_time, '.6f')}' sec"
            )
            is_valid = self._validate_train_json(json_data)
            if not is_valid:
                logger.debug(f"Invalid metadata for '{artifacts_folder}'")
                json_data = None
        return json_data

    def _validate_sort(self, sort: Literal["desc", "asc"]):
        if sort not in ["desc", "asc"]:
            raise ValueError(f"Invalid sort value: {sort}")

    def _get_file_infos(self):
        return self._api.file.list(self._team_id, self._framework_folder, return_type="fileinfo")

    def _group_files_by_folder(self, file_infos: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        folders = defaultdict(list)
        for file_info in file_infos:
            artifacts_folder = dirname(file_info.path)
            if self.is_valid_artifacts_path(artifacts_folder):
                folders[artifacts_folder].append(file_info)
        return folders

    def _validate_train_json(self, train_json) -> bool:
        if not isinstance(train_json, dict):
            # Invalid train_json format
            return False
        required_fields = [
            "app_name",
            "task_id",
            "artifacts_folder",
            "session_link",
            "task_type",
            "project_name",
            "checkpoints",
        ]
        if self.app_name == "Train MMDetection" or self.app_name == "Train MMDetection 3.0":
            required_fields.append("config_path")

        for field in required_fields:
            if field not in train_json:
                # Missing 'field' in train_json
                return False
            else:
                value = train_json.get(field)
                if value is None:
                    logger.debug(f"Field '{field}' is None")
                    # 'field' is None
                    return False
        return True

    def _create_train_infos(self, folders):
        train_infos = []

        def process_folder(artifacts_folder, file_infos):
            metadata_path = join(artifacts_folder, self._metadata_file_name)
            file_paths = [file_info.path for file_info in file_infos]
            train_json = self._get_train_json(
                artifacts_folder, metadata_path, file_infos, file_paths
            )
            if train_json:
                return TrainInfo(**train_json)
            return None

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_folder, folder, infos): folder
                for folder, infos in folders.items()
            }
            for future in as_completed(futures):
                train_info = future.result()
                if train_info:
                    train_infos.append(train_info)

        return train_infos

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[TrainInfo]:
        """
        Return list of custom training infos

        :param sort: The sort order, either "desc" or "asc". Default is "desc".
        :type sort: Literal["desc", "asc"]
        :return: The list of custom training artifact infos.
        :rtype: List[TrainInfo]
        """
        self._validate_sort(sort)
        start_time = time()
        parsed_infos = self._get_file_infos()
        folders = self._group_files_by_folder(parsed_infos)

        with ThreadPoolExecutor() as executor:
            future = executor.submit(self._create_train_infos, folders)
            train_infos = future.result()

        end_time = time()
        train_infos = self.sort_train_infos(train_infos, sort)
        logger.debug(f"Listing time: '{format(end_time - start_time, '.6f')}' sec")
        return train_infos

    def convert_train_to_experiment_info(
        self, train_info: TrainInfo
    ) -> Union['ExperimentInfo', None]:
        try:
            checkpoints = []
            for chk in train_info.checkpoints:
                if self.weights_folder:
                    checkpoints.append(join(self.weights_folder, chk.name))
                else:
                    checkpoints.append(chk.name)

            best_checkpoint = next(
                (chk.name for chk in train_info.checkpoints if "best" in chk.name), None
            )
            if not best_checkpoint and checkpoints:
                best_checkpoint = get_file_name_with_ext(checkpoints[-1])

            task_info = self._api.task.get_info_by_id(train_info.task_id)
            workspace_id = task_info["workspaceId"]

            project = self._api.project.get_info_by_name(workspace_id, train_info.project_name)
            project_id = project.id if project else None

            model_files = {}
            if train_info.config_path:
                model_files["config"] = self.get_config_path(train_info.artifacts_folder).replace(
                    train_info.artifacts_folder, ""
                )

            input_datetime = task_info["startedAt"]
            parsed_datetime = datetime.strptime(input_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
            date_time = parsed_datetime.strftime("%Y-%m-%d %H:%M:%S")

            experiment_info_data = {
                "experiment_name": f"{self.framework_name} experiment",
                "framework_name": self.framework_name,
                "model_name": f"{self.framework_name} model",
                "task_type": train_info.task_type,
                "project_id": project_id,
                "task_id": train_info.task_id,
                "model_files": model_files,
                "checkpoints": checkpoints,
                "best_checkpoint": best_checkpoint,
                "artifacts_dir": train_info.artifacts_folder,
                "datetime": date_time,
            }

            experiment_info_fields = {
                field.name
                for field in ExperimentInfo.__dataclass_fields__.values()  # pylint: disable=no-member
            }
            for field in experiment_info_fields:
                if field not in experiment_info_data:
                    experiment_info_data[field] = None
            return ExperimentInfo(**experiment_info_data)
        except Exception as e:
            logger.debug(f"Failed to build experiment info: {e}")
            return None

    def get_list_experiment_info(
        self, sort: Literal["desc", "asc"] = "desc"
    ) -> List['ExperimentInfo']:
        train_infos = self.get_list(sort)

        # Sync version
        # Uncomment for debug
        # experiment_infos = [
        #     convert_traininfo_to_experimentinfo(api, framework_cls, train_info)
        #     for train_info in train_infos
        # ]
        # return experiment_infos

        # Async version
        with ThreadPoolExecutor() as executor:
            experiment_infos = list(
                executor.map(
                    lambda t: self.convert_train_to_experiment_info(t),
                    train_infos,
                )
            )
        return [info for info in experiment_infos if info is not None]

    def get_available_task_types(self) -> List[str]:
        """
        Get available task types.

        :return: The list of available task types.
        :rtype: List[str]
        """
        return self._available_task_types

    def get_info_by_artifacts_dir(
        self,
        artifacts_dir: str,
        return_type: Literal["train_info", "experiment_info"] = "train_info",
    ) -> Union[TrainInfo, 'ExperimentInfo', None]:
        """
        Get training info by artifacts directory.

        :param artifacts_dir: The artifacts directory.
        :type artifacts_dir: str
        :param return_type: The return type, either "train_info" or "experiment_info". Default is "experiment_info".
        :type return_type: Literal["train_info", "experiment_info"]
        :return: The training info.
        :rtype: TrainInfo
        """
        for train_info in self.get_list():
            if train_info.artifacts_folder == artifacts_dir:
                if return_type == "train_info":
                    return train_info
                else:
                    return self.convert_train_to_experiment_info(train_info)

    # load_custom_checkpoint
    # inference
    # fix docstrings :param: x -> :param x:
