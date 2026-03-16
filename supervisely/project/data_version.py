import io
import os
import tarfile
import tempfile
import time
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple, Union

import requests
import zstd

from supervisely._utils import logger
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.api.project_api import ProjectInfo, ProjectType
from supervisely.io import json
from supervisely.io.fs import remove_dir
from supervisely.project.versioning.common import (
    DEFAULT_IMAGE_SCHEMA_VERSION,
    DEFAULT_VIDEO_SCHEMA_VERSION,
    DEFAULT_VOLUME_SCHEMA_VERSION,
    HIDDEN_WORKSPACE_NAME,
    PREVIEW_DESCRIPTION_TEMPLATE,
    PREVIEW_NAME_TEMPLATE,
    update_custom_data_with_version_preview,
)


class VersionInfo(NamedTuple):
    """
    Object with parameters from Supervisely that describes the version of the project.
    """

    id: int
    project_id: int
    created_by: int
    team_file_id: int
    version: int
    description: str
    status: str
    created_at: str
    updated_at: str
    project_updated_at: str
    team_id: int
    name: str
    preview_project_id: Optional[int] = None


class DataVersion(ModuleApiBase):
    """
    Class for managing project versions.
    This class provides methods for creating, restoring, and managing project versions.
    """

    PROJECT_NAME_TEMPLATE = "{project_name}, from ver. {version_num}"
    PROJECT_DESC_TEMPLATE = (
        "Restored from version {version_num}. "
        "Source project ID: {project_id}, version ID: {version_id}"
    )

    def __init__(self, api):
        """
        Class for managing project versions.
        """
        from supervisely import Api

        self._api: Api = api
        self.__storage_dir: str = "/system/versions/"
        self.__version_format: str = DEFAULT_IMAGE_SCHEMA_VERSION
        self.project_info = None
        self.project_dir = None
        self.versions_path = None
        self.versions = None
        self._batch_size = None
        self._local_binary_path = None

    @staticmethod
    def info_sequence():
        """
        NamedTuple VersionInfo with API Fields containing information about Project Version.

        """

        return [
            ApiField.ID,
            ApiField.PROJECT_ID,
            ApiField.CREATED_BY_ID,
            ApiField.TEAM_FILE_ID,
            ApiField.VERSION,
            ApiField.DESCRIPTION,
            ApiField.STATUS,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.PROJECT_UPDATED_AT,
            ApiField.TEAM_ID,
            ApiField.NAME,
            ApiField.PREVIEW_PROJECT_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **VersionInfo**.
        """
        return "VersionInfo"

    @property
    def project_cls(self):
        from supervisely.project import (
            Project,
            ProjectType,
            VideoProject,
            VolumeProject,
        )

        if self.project_info is None:
            raise ValueError("Project info is not initialized. Call 'initialize' method first.")

        self._batch_size = None
        project_type = self.project_info.type
        if project_type == ProjectType.IMAGES.value:
            self.__version_format = DEFAULT_IMAGE_SCHEMA_VERSION
            self._batch_size = 200
            return Project
        elif project_type == ProjectType.VIDEOS.value:
            self.__version_format = DEFAULT_VIDEO_SCHEMA_VERSION
            self._batch_size = 50
            return VideoProject
        elif project_type == ProjectType.VOLUMES.value:
            self.__version_format = DEFAULT_VOLUME_SCHEMA_VERSION
            self._batch_size = 50
            return VolumeProject
        else:
            raise ValueError(f"Unsupported project type: {project_type}")

    def initialize(self, project_info: Union[ProjectInfo, int]):
        """
        Initialize project versions.

        :param project_info: Project info object or project ID
        :type project_info: Union[:class:`~supervisely.api.project_api.ProjectInfo`, int]
        """
        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)
        self.project_info: ProjectInfo = project_info
        self.project_dir: str = os.path.join(self.__storage_dir, str(self.project_info.id))
        self.versions_path: str = os.path.join(self.project_dir, "versions.json")
        self.versions: dict = self.get_map(self.project_info, do_initialization=False)
        if self.project_info.version is None:
            self._create_warning_system_file()

    def get_list(self, project_id: int, filters: Optional[List] = None) -> List[VersionInfo]:
        """
        Get list of project versions.

        :param project_id: Project ID
        :type project_id: int
        :param filters: Filters
        :type filters: Optional[List]
        :returns: List of project versions
        :rtype: List[:class:`~supervisely.project.data_version.VersionInfo`]
        """
        data = {ApiField.PROJECT_ID: project_id}
        if filters:
            data[ApiField.FILTER] = filters

        return self.get_list_all_pages("projects.versions.list", data)

    def get_info_by_id(self, project_id: int, version_id: int) -> Optional[VersionInfo]:
        """
        Get project version information by version ID.

        :param project_id: Project ID
        :type project_id: int
        :param version_id: Version ID
        :type version_id: int
        :returns: Project version information
        :rtype: Optional[:class:`~supervisely.project.data_version.VersionInfo`]
        """
        versions = self.get_list(
            project_id,
            filters=[
                {ApiField.FIELD: ApiField.ID, ApiField.OPERATOR: "=", ApiField.VALUE: version_id}
            ],
        )
        return versions[0] if versions else None

    def get_info_by_number(self, project_id: int, version_num: int) -> Optional[VersionInfo]:
        """
        Get project version information by version number.

        :param project_id: Project ID
        :type project_id: int
        :param version_num: Version number
        :type version_num: int
        :returns: Project version information
        :rtype: Optional[:class:`~supervisely.project.data_version.VersionInfo`]
        """
        versions = self.get_list(
            project_id,
            filters=[
                {
                    ApiField.FIELD: ApiField.VERSION,
                    ApiField.OPERATOR: "=",
                    ApiField.VALUE: version_num,
                }
            ],
        )
        return versions[0] if versions else None

    def get_id_by_number(self, project_id: int, version_num: int) -> int:
        """
        Get version ID by version number.

        :param project_id: Project ID
        :type project_id: int
        :param version_num: Version number
        :type version_num: int
        :returns: Version ID
        :rtype: int or None
        """
        filter = [
            {
                ApiField.FIELD: ApiField.VERSION,
                ApiField.OPERATOR: "=",
                ApiField.VALUE: int(version_num),
            }
        ]
        versions = self.get_list(project_id, filters=filter)
        if len(versions) > 0:
            return versions[0].id
        return None

    def get_map(self, project_info: Union[ProjectInfo, int], do_initialization: bool = True):
        """
        Get project versions map from storage.

        :param project_info: Project info object or project ID
        :type project_info: Union[:class:`~supervisely.api.project_api.ProjectInfo`, int]
        :param do_initialization: Initialize project versions. Set to False for internal use.
        :type do_initialization: bool
        :returns: Project versions
        :rtype: dict
        """
        if do_initialization:
            self.initialize(project_info)

        try:
            versions = self._api.file.get_json_file_content(
                self.project_info.team_id, self.versions_path
            )
            return versions or {}
        except FileNotFoundError:
            # versions = {"format": self.__version_format}
            return {}

    def set_map(self, project_info: Union[ProjectInfo, int], initialize: bool = True):
        """
        Save project versions map to storage.

        :param project_info: Project info object or project ID
        :type project_info: Union[:class:`~supervisely.api.project_api.ProjectInfo`, int]
        :param initialize: Initialize project versions. Set to False for internal use.
        :type initialize: bool
        :returns: None
        """

        if initialize:
            self.initialize(project_info)
        if "format" not in self.versions:
            self.versions["format"] = self.__version_format
        temp_dir = tempfile.mkdtemp()
        local_versions = os.path.join(temp_dir, "versions.json")
        json.dump_json_file(self.versions, local_versions)
        file_info = self._api.file.upload(
            self.project_info.team_id, local_versions, self.versions_path
        )
        if file_info is None:
            raise RuntimeError("Failed to save versions")

        remove_dir(temp_dir)

    def create(
        self,
        project_info: Union[ProjectInfo, int],
        version_title: Optional[str] = None,
        version_description: Optional[str] = None,
        enable_preview: bool = False,
    ) -> int:
        """
        Create a new project version.
        Returns the ID of the new version.
        If the project is already on the latest version, returns the latest version ID.
        If the project version cannot be created, returns None.

        :param project_info: Project info object or project ID
        :type project_info: Union[:class:`~supervisely.api.project_api.ProjectInfo`, int]
        :param version_title: Version title
        :type version_title: Optional[str]
        :param version_description: Version description
        :type version_description: Optional[str]
        :param enable_preview: Enable preview flag that creates clone of the project to hidden workspace making version data available immediately. This option can be used to
        :type enable_preview: bool
        :returns: Version ID
        :rtype: int
        """
        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)

        if (
            "app.supervise.ly" in self._api.server_address
            or "app.supervisely.com" in self._api.server_address
        ):
            if self._api.team.get_info_by_id(project_info.team_id).usage.plan == "free":
                logger.warning(
                    "Project versioning is not available for teams with Free plan. Please upgrade to Pro to enable versioning."
                )
                return None

        self.initialize(project_info)
        path = self._generate_save_path()
        latest = self._get_latest_id()
        version_num = int(self.versions[str(latest)]["number"]) + 1 if latest else 1
        try:
            version_id, commit_token = self.reserve(project_info.id)
        except Exception as e:
            logger.error(f"Failed to reserve version. Exception: {e}")
            return None
        if version_id is None and commit_token is None:
            return latest
        try:

            file_info = self._compress_and_upload(path, preserve_local_binary=enable_preview)

            self.versions[version_id] = {
                "path": path,
                "updated_at": project_info.updated_at,
                "previous": latest,
                "number": version_num,
            }
            # if enable_preview and preview_project_info is not None:
            #     self.versions[version_id]["preview"] = preview_project_info.id
            self.versions["latest"] = version_id
            self.set_map(project_info, initialize=False)
            self.commit(
                version_id,
                commit_token,
                project_info.updated_at,
                file_info.id,
                title=version_title,
                description=version_description,
            )
            if enable_preview:
                if project_info.type in [ProjectType.VIDEOS.value, ProjectType.IMAGES.value]:
                    preview_project_info = self.enable_preview(
                        project=project_info,
                        version_id=version_id,
                        local_binary_path=self._local_binary_path,
                    )
                    if self._local_binary_path is not None:
                        remove_dir(os.path.dirname(self._local_binary_path))
                        self._local_binary_path = None
                    self.versions[str(version_id)]["preview"] = preview_project_info.id
                    self.set_map(project_info, initialize=False)
                else:
                    logger.warning(
                        f"Preview is not supported for project type {project_info.type}. Creating version without preview."
                    )
            return version_id
        except Exception as e:
            if self.cancel_reservation(version_id, commit_token):
                logger.error(f"Version creation failed. Reservation was cancelled. Exception: {e}")
            else:
                logger.error(
                    f"Failed to cancel reservation when handling exception. You can cancel your reservation on the web under the Versions tab of the project. Exception: {e}"
                )
            return None

    def commit(
        self,
        version_id: int,
        commit_token: str,
        updated_at: str,
        file_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        preview_project_id: Optional[int] = None,
    ):
        """
        Commit project version.
        This method is used to finalize the version creation process.
        Requires active reservation.
        You must call this method after creating project version backup and setting version map

        :param version_id: Version ID
        :type version_id: int
        :param commit_token: Commit token
        :type commit_token: str
        :param updated_at: Updated at timestamp
        :type updated_at: str
        :param file_id: File ID
        :type file_id: int
        :param title: Version title
        :type title: Optional[str]
        :param description: Version description
        :type description: Optional[str]
        :param preview_project_id: ID of the cloned project that will be used to preview version data.
        :type preview_project_id: Optional[int]
        :returns: None
        """
        payload = {
            ApiField.ID: version_id,
            ApiField.COMMIT_TOKEN: commit_token,
            ApiField.PROJECT_UPDATED_AT: updated_at,
            ApiField.TEAM_FILE_ID: file_id,
        }
        if preview_project_id is not None:
            payload[ApiField.PREVIEW_PROJECT_ID] = preview_project_id
        if title:
            payload[ApiField.TITLE] = title
        if description:
            payload[ApiField.DESCRIPTION] = description

        response = self._api.post("projects.versions.commit", payload)
        commit_info = response.json()
        if not commit_info.get("success"):
            raise RuntimeError("Failed to commit version")

    def reserve(self, project_id: int, retries: int = 6) -> Tuple[int, str]:
        """
        Reserve project version.
        This method is used before backing up a version to prevent another attempt to create a version at the same time.
        The first delay of retry is 2 seconds, which doubles with each subsequent attempt.

        :param project_id: Project ID
        :type project_id: int
        :param retries: Number of attempts to reserve version
        :type retries: int
        :returns: Version ID and commit token
        :rtype: Tuple[int, str]
        """
        retry_delay = 2  # seconds
        max_delay = retry_delay * 2**retries

        while True:
            try:
                response = self._api.post(
                    "projects.versions.reserve", {ApiField.PROJECT_ID: project_id}
                )
                reserve_info = response.json()
                return reserve_info.get(ApiField.ID), reserve_info.get(ApiField.COMMIT_TOKEN)

            except requests.exceptions.HTTPError as e:
                details = {}
                if e.response is not None:
                    try:
                        details = (e.response.json() or {}).get("details", {})  # type: ignore[union-attr]
                    except Exception:
                        details = {}

                if details.get("useExistingVersion"):
                    version_id = details.get("version", {}).get("id")
                    version = details.get("version", {}).get("version")
                    logger.info(
                        f"No changes to the project since the last version '{version}' with ID '{version_id}'"
                    )
                    return (None, None)

                message = (details.get("message") or "").lower()
                if "is already committing" in message:
                    if retry_delay >= max_delay:
                        raise RuntimeError(
                            "Failed to reserve version. Another process is already committing a version. Maximum number of attempts reached."
                        )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                raise

    def cancel_reservation(self, version_id: int, commit_token: str):
        """
        Cancel version reservation for a project.

        :param version_id: Version ID
        :type version_id: int
        :param commit_token: Commit token
        :type commit_token: str
        :returns: True if reservation was cancelled, False otherwise
        """
        response = self._api.post(
            "projects.versions.cancel-reservation",
            {ApiField.ID: version_id, ApiField.COMMIT_TOKEN: commit_token},
        )
        reserve_info = response.json()
        return True if reserve_info.get("success") else False

    def restore(
        self,
        project_info: Union[ProjectInfo, int],
        version_id: Optional[int] = None,
        version_num: Optional[int] = None,
        skip_missed_entities: bool = False,
        workspace_id: Optional[int] = None,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        local_binary_path: Optional[str] = None,
    ) -> ProjectInfo:
        """
        Restore project to a specific version.
        Version can be specified by ID or number.

        :param project_info: Project info object or project ID
        :type project_info: Union[:class:`~supervisely.api.project_api.ProjectInfo`, int]
        :param version_id: Version ID
        :type version_id: Optional[int]
        :param version_num: Version number
        :type version_num: Optional[int]
        :param skip_missed_entities: Skip missed Images
        :type skip_missed_entities: bool, default False
        :param workspace_id: Workspace ID where the restored project will be created. If None, the project will be restored to the same workspace.
        :type workspace_id: Optional[int]
        :param project_name: Name of the restored project. If None, a default name will be used.
        :type project_name: Optional[str]
        :param project_description: Description of the restored project. If None, a default description will be used.
        :type project_description: Optional[str]
        :param local_binary_path: Path to the local binary file to restore from. If None, will download from the server.
        :type local_binary_path: Optional[str]
        :returns: Project info object of the restored project
        :rtype: :class:`~supervisely.api.project_api.ProjectInfo` or None
        """
        if version_id is None and version_num is None:
            raise ValueError("Either version_id or version_num must be provided")

        self.initialize(project_info)

        if version_num:
            version_id = None
            for key, value in self.versions.items():
                # pylint: disable=no-member
                if isinstance(value, dict) and value.get("number") == version_num:
                    version_id = key
                    break
            if version_id is None:
                raise ValueError(f"Version {version_num} does not exist")
        else:
            if str(version_id) not in self.versions:
                raise ValueError(f"Version {version_id} does not exist")
            version_num = self.versions[str(version_id)]["number"]

        if project_name is None:
            dst_project_name = self.PROJECT_NAME_TEMPLATE.format(
                project_name=self.project_info.name, version_num=version_num
            )
        else:
            dst_project_name = project_name

        if project_description is None:
            dst_project_desc = self.PROJECT_DESC_TEMPLATE.format(
                version_num=version_num,
                project_id=self.project_info.id,
                version_id=version_id,
            )
        else:
            dst_project_desc = project_description

        # updated_at = self.versions[str(version_id)]["updated_at"]
        backup_files = self.versions[str(version_id)]["path"]

        # turn off this check for now (treating this as a project clone operation)
        # if updated_at == self.project_info.updated_at:
        #     logger.warning(
        #         f"Project is already on version {version_num} with the same updated_at timestamp"
        #     )
        #     return

        if backup_files is None:
            logger.warning(
                f"Project can't be restored to version {version_num} because it doesn't have restore point."
            )
            return

        if local_binary_path is not None:
            with open(local_binary_path, "rb") as f:
                bin_io = io.BytesIO(f.read())
        else:
            bin_io = self._download_and_extract(backup_files)

        new_project_info = self.project_cls.upload_bin(
            self._api,
            bin_io,
            workspace_id=self.project_info.workspace_id if workspace_id is None else workspace_id,
            project_name=dst_project_name,
            project_description=dst_project_desc,
            skip_missed=skip_missed_entities,
        )
        return new_project_info

    def get_or_create_versions_workspace(self, team_id: int, description: str = "") -> int:
        """
        Get or create a hidden workspace for storing preview project versions for a team.

        :param team_id: Team ID
        :type team_id: int
        :param description: Workspace description
        :type description: str
        :returns: Workspace ID
        :rtype: int
        """
        workspace_info = self._api.workspace.get_info_by_name(team_id, HIDDEN_WORKSPACE_NAME)

        if workspace_info is not None:
            return workspace_info.id

        new_workspace = self._api.workspace.create(
            team_id=team_id,
            name=HIDDEN_WORKSPACE_NAME,
            description=description,
            hidden=True,
        )
        return new_workspace.id

    def update(
        self,
        version_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        preview_project_id: Optional[int] = None,
    ):
        """
        Update version information such as name, description or link preview project to the version.

        ATTENTION: Do not use this parameter to link a regular project as it can cause issues with version data consistency. This parameter is intended to link a cloned project in the hidden workspace that is created when enabling preview for the version.
        Only versions of projects with types VIDEO and IMAGE are supported for preview.

        :param version_id: Version ID
        :type version_id: int
        :param name: New name
        :type name: Optional[str]
        :param description: New description
        :type description: Optional[str]
        :param preview_project_id: Preview project ID to link to the version. This project will be used to preview version data. Only versions of projects with types VIDEO and IMAGE are supported for preview.
        :type preview_project_id: Optional[int]
        :returns: None
        :rtype: None
        """
        if name is None and description is None and preview_project_id is None:
            raise ValueError(
                "At least one of name, description or preview_project_id must be provided"
            )

        payload = {
            ApiField.ID: version_id,
        }
        if name is not None:
            payload[ApiField.NAME] = name
        if description is not None:
            payload[ApiField.DESCRIPTION] = description
        if preview_project_id is not None:
            payload[ApiField.PREVIEW_PROJECT_ID] = preview_project_id
        response = self._api.post("projects.versions.update", payload)
        update_info = response.json()
        if not update_info.get("success"):
            raise RuntimeError("Failed to update version information")

    def enable_preview(
        self,
        project: Union[int, ProjectInfo],
        version_id: int,
        overwrite: bool = False,
        local_binary_path: Optional[str] = None,
    ) -> ProjectInfo:
        """
        Enable preview for the version by creating a snapshot project and linking it to the version.
        If the snapshot project already exists and overwrite is False, returns the existing snapshot project ID.

        ATTENTION: This method works only for committed versions with successfully uploaded version data.

        :param project: Source project ID or ProjectInfo object
        :type project: Union[int, ProjectInfo]
        :param version_id: Version ID
        :type version_id: int
        :param overwrite: Whether to overwrite existing snapshot project if it exists
        :type overwrite: bool
        :param local_binary_path: Path to a local version.bin file to use instead of downloading from server.
        :type local_binary_path: Optional[str]
        :returns: Preview snapshot project information
        :rtype: ProjectInfo
        """

        if isinstance(project, int):
            project_id = project
            project_info = self._api.project.get_info_by_id(project_id)
        elif isinstance(project, ProjectInfo):
            project_id = project.id
            project_info = project
        else:
            raise ValueError(f"Invalid object type: {type(project)}. Must be int or ProjectInfo.")

        if project_info.type not in [ProjectType.VIDEOS.value, ProjectType.IMAGES.value]:
            raise ValueError(f"Preview is not supported for project type {project_info.type}")

        version_info = self.get_info_by_id(project_id, version_id)

        if version_info is None:
            raise ValueError(f"Version with ID {version_id} does not exist")

        logger.info(
            f"Enabling preview for version ID: {version_id} of project ID: {project_info.id} with overwrite={overwrite}"
        )

        if version_info.preview_project_id is not None and not overwrite:
            logger.info(
                f"Preview snapshot project with ID {version_info.preview_project_id} already exists for version {version_id}. Returning existing snapshot project information."
            )
            return self._api.project.get_info_by_id(version_info.preview_project_id)

        preview_project_info = self.restore(
            project_info=project_info,
            version_id=version_id,
            workspace_id=self.get_or_create_versions_workspace(team_id=project_info.team_id),
            project_name=PREVIEW_NAME_TEMPLATE.format(
                project_name=project_info.name, version_num=version_info.version
            ),
            project_description=PREVIEW_DESCRIPTION_TEMPLATE.format(
                version_num=version_info.version, project_id=project_info.id, version_id=version_id
            ),
            local_binary_path=local_binary_path,
        )
        custom_data = preview_project_info.custom_data
        if custom_data is None:
            custom_data = self._api.project.get_custom_data(preview_project_info.id) or {}

        self.update(version_id, preview_project_id=preview_project_info.id)
        self._api.project.set_read_only(preview_project_info.id, True)
        # refresh project info to get updated_at timestamp after restore
        preview_project_info = self._api.project.get_info_by_id(preview_project_info.id)
        custom_data = update_custom_data_with_version_preview(
            custom_data=custom_data,
            version_id=version_id,
            source_project_id=project_info.id,
            preview_created_at=preview_project_info.updated_at,
        )
        self._api.project.update_custom_data(
            id=preview_project_info.id, data=custom_data, silent=True
        )

        return preview_project_info

    def _create_warning_system_file(self):
        """
        Create a file in the system directory to indicate that you cannot manually modify its contents.

        Path = /system/DO_NOT_DELETE_ANYTHING_HERE.txt

        """
        warning_file = "/system/DO_NOT_DELETE_ANYTHING_HERE.txt"
        if not self._api.file.exists(self.project_info.team_id, warning_file, recursive=False):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open(temp_file.name, "w") as f:
                f.write("This directory is managed by Supervisely. Do not modify its contents.")
            self._api.file.upload(self.project_info.team_id, temp_file.name, warning_file)

    def _download_and_extract(self, path: str) -> io.BytesIO:
        """
        Download and extract version data to memory.

        :param path: Path to the version file
        :type path: str
        :returns: Binary IO object with extracted file
        :rtype: io.BytesIO
        """
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "download.tar.zst")
        try:
            self._api.file.download(self.project_info.team_id, path, local_path)
            # Stream-decompress and stream-read tar to avoid loading the whole archive in memory.
            try:
                dctx = zstd.ZstdDecompressor()
                with open(local_path, "rb") as zst_f:
                    with dctx.stream_reader(zst_f) as reader:
                        with tarfile.open(fileobj=reader, mode="r|") as tar:
                            for member in tar:
                                if member.name == "version.bin":
                                    file = tar.extractfile(member)
                                    if not file:
                                        raise RuntimeError("version.bin not found in the archive")
                                    return io.BytesIO(file.read())
                            raise RuntimeError("version.bin not found in the archive")
            except Exception:
                # Fallback: one-shot decompress
                with open(local_path, "rb") as zst_f:
                    decompressed_data = zstd.decompress(zst_f.read())
                with tarfile.open(fileobj=io.BytesIO(decompressed_data), mode="r") as tar:
                    file = tar.extractfile("version.bin")
                    if not file:
                        raise RuntimeError("version.bin not found in the archive")
                    return io.BytesIO(file.read())
        except Exception as e:
            raise RuntimeError(f"Failed to extract version: {e}")
        finally:
            remove_dir(temp_dir)

    def _generate_save_path(self):
        """
        Generate a path for the new version archive where it will be saved in the Team Files.
        Archive format: {timestamp}.tar.zst

        :returns: Path for the new version archive
        :rtype: str
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = os.path.join(self.project_dir, timestamp + ".tar.zst")
        return path

    def _get_latest_id(self):
        """
        Get the ID of the latest version from the versions map (versions.json).
        """
        latest = self.versions.get("latest", None)
        if not latest:
            return None
        return latest

    def _compress_and_upload(self, path: str, preserve_local_binary: bool = False) -> dict:
        """
        Save project in compressed binary format to the Team Files.
        Binary file name: version.bin

        :param path: Destination path where the version archive will be saved in the Team Files
        :type path: str
        :param preserve_local_binary: Whether to preserve the local copy of the binary version.
            If False, the local copy will be deleted after uploading.
        :type preserve_local_binary: bool
        :returns: File info
        :rtype: dict
        """
        temp_dir = tempfile.mkdtemp()
        data = None
        version_bin_path = None
        try:
            data = self.project_cls.download_bin(
                self._api, self.project_info.id, batch_size=self._batch_size, return_bytesio=True
            )

            version_bin_path = os.path.join(temp_dir, "version.bin")
            data.seek(0)
            with open(version_bin_path, "wb") as f:
                f.write(data.read())

            # Set the path for future use if preserve_local_binary is True
            if preserve_local_binary:
                self._local_binary_path = version_bin_path

            zst_archive_path = os.path.join(temp_dir, "download.tar.zst")
            file_size = os.path.getsize(version_bin_path)

            # Stream compress from file to avoid loading whole archive in memory
            try:
                cctx = zstd.ZstdCompressor()
                with open(zst_archive_path, "wb") as zst_f:
                    try:
                        stream = cctx.stream_writer(zst_f, closefd=False)
                    except TypeError:
                        stream = cctx.stream_writer(zst_f)
                    with stream as compressor:
                        with tarfile.open(fileobj=compressor, mode="w|") as tar:
                            info = tarfile.TarInfo(name="version.bin")
                            info.size = file_size
                            with open(version_bin_path, "rb") as f:
                                tar.addfile(tarinfo=info, fileobj=f)
            except Exception:
                # Fallback: build tar in memory + one-shot compress
                tar_data = io.BytesIO()
                with tarfile.open(fileobj=tar_data, mode="w") as tar:
                    info = tarfile.TarInfo(name="version.bin")
                    info.size = file_size
                    with open(version_bin_path, "rb") as f:
                        tar.addfile(tarinfo=info, fileobj=f)
                tar_data.seek(0)
                with open(zst_archive_path, "wb") as zst_f:
                    zst_f.write(zstd.compress(tar_data.read()))

            file_info = self._api.file.upload(self.project_info.team_id, zst_archive_path, path)
            return file_info
        finally:
            if data is not None:
                try:
                    data.close()
                except Exception:
                    pass
            if not preserve_local_binary:
                remove_dir(temp_dir)
