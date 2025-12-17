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
from supervisely.api.project_api import ProjectInfo
from supervisely.io import json
from supervisely.io.fs import remove_dir, silent_remove


class VersionInfo(NamedTuple):
    """
    Object with image parameters from Supervisely that describes the version of the project.
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


class DataVersion(ModuleApiBase):
    """
    Class for managing project versions.
    This class provides methods for creating, restoring, and managing project versions.
    """

    def __init__(self, api):
        """
        Class for managing project versions.
        """
        from supervisely import Api

        self._api: Api = api
        self.__storage_dir: str = "/system/versions/"
        self.__version_format: str = "v1.0.0"
        self.project_info = None
        self.project_dir = None
        self.versions_path = None
        self.versions = None

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
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **VersionInfo**.
        """
        return "VersionInfo"

    def initialize(self, project_info: Union[ProjectInfo, int]):
        """
        Initialize project versions.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
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
        :return: List of project versions
        :rtype: List[VersionInfo]
        """
        data = {ApiField.PROJECT_ID: project_id}
        if filters:
            data[ApiField.FILTER] = filters

        return self.get_list_all_pages("projects.versions.list", data)

    def get_id_by_number(self, project_id: int, version_num: int) -> int:
        """
        Get version ID by version number.

        :param project_id: Project ID
        :type project_id: int
        :param version_num: Version number
        :type version_num: int
        :return: Version ID
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

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param do_initialization: Initialize project versions. Set to False for internal use.
        :type do_initialization: bool
        :return: Project versions
        :rtype: dict
        """
        if do_initialization:
            self.initialize(project_info)

        try:
            versions = self._api.file.get_json_file_content(
                self.project_info.team_id, self.versions_path
            )
            versions = versions if versions else {}
        except FileNotFoundError:
            versions = {"format": self.__version_format}
        return versions

    def set_map(self, project_info: Union[ProjectInfo, int], initialize: bool = True):
        """
        Save project versions map to storage.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param initialize: Initialize project versions. Set to False for internal use.
        :type initialize: bool
        :return: None
        """

        if initialize:
            self.initialize(project_info)
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
    ) -> int:
        """
        Create a new project version.
        Returns the ID of the new version.
        If the project is already on the latest version, returns the latest version ID.
        If the project version cannot be created, returns None.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param version_title: Version title
        :type version_title: Optional[str]
        :param version_description: Version description
        :type version_description: Optional[str]
        :return: Version ID
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
        try:
            version_id, commit_token = self.reserve(project_info.id)
        except Exception as e:
            logger.error(f"Failed to reserve version. Exception: {e}")
            return None
        if version_id is None and commit_token is None:
            return latest
        try:
            file_info = self._compress_and_upload(path)
            self.versions[version_id] = {
                "path": path,
                "updated_at": project_info.updated_at,
                "previous": latest,
                "number": int(self.versions[str(latest)]["number"]) + 1 if latest else 1,
            }
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
        :return: None
        """
        body = {
            ApiField.ID: version_id,
            ApiField.COMMIT_TOKEN: commit_token,
            ApiField.PROJECT_UPDATED_AT: updated_at,
            ApiField.TEAM_FILE_ID: file_id,
        }
        if title:
            body[ApiField.TITLE] = title
        if description:
            body[ApiField.DESCRIPTION] = description

        response = self._api.post("projects.versions.commit", body)
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
        :return: Version ID and commit token
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
                if e.response.json().get("details", {}).get("useExistingVersion"):
                    version_id = e.response.json().get("details", {}).get("version").get("id")
                    version = e.response.json().get("details", {}).get("version").get("version")
                    logger.info(
                        f"No changes to the project since the last version '{version}' with ID '{version_id}'"
                    )
                    return (None, None)
                elif "is already committing" in e.response.json().get("details", {}).get("message"):
                    if retry_delay >= max_delay:
                        raise RuntimeError(
                            "Failed to reserve version. Another process is already committing a version. Maximum number of attempts reached."
                        )
                    version = e.response.json().get("details", {}).get("version").get("version")
                    time.sleep(retry_delay)
                    retry_delay *= 2

    def cancel_reservation(self, version_id: int, commit_token: str):
        """
        Cancel version reservation for a project.

        :param version_id: Version ID
        :type version_id: int
        :param commit_token: Commit token
        :type commit_token: str
        :return: True if reservation was cancelled, False otherwise
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
    ) -> ProjectInfo:
        """
        Restore project to a specific version.
        Version can be specified by ID or number.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param version_id: Version ID
        :type version_id: Optional[int]
        :param version_num: Version number
        :type version_num: Optional[int]
        :param skip_missed_entities: Skip missed Images
        :type skip_missed_entities: bool, default False
        :return: ProjectInfo object of the restored project
        :rtype: ProjectInfo or None
        """
        from supervisely.project.project import Project

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
        updated_at = self.versions[str(version_id)]["updated_at"]
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

        bin_io = self._download_and_extract(backup_files)
        new_project_info = Project.upload_bin(
            self._api,
            bin_io,
            self.project_info.workspace_id,
            skip_missed=skip_missed_entities,
        )
        return new_project_info

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
        :return: Binary IO object with extracted file
        :rtype: io.BytesIO
        """
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "download.tar.zst")
        try:
            self._api.file.download(self.project_info.team_id, path, local_path)
            with open(local_path, "rb") as zst:
                decompressed_data = zstd.decompress(zst.read())
            with tarfile.open(fileobj=io.BytesIO(decompressed_data)) as tar:
                file = tar.extractfile("version.bin")
                if not file:
                    raise RuntimeError("version.bin not found in the archive")
                data = file.read()
                bin_io = io.BytesIO(data)
                return bin_io
        except Exception as e:
            raise RuntimeError(f"Failed to extract version: {e}")
        finally:
            remove_dir(temp_dir)

    def _generate_save_path(self):
        """
        Generate a path for the new version archive where it will be saved in the Team Files.
        Archive format: {timestamp}.tar.zst

        :return: Path for the new version archive
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

    def _compress_and_upload(self, path: str) -> dict:
        """
        Save project in binary format in archive to the Team Files.
        Binary file name: version.bin

        :param changes: Changes between current and previous version
        :type changes: bool
        :return: File info
        :rtype: dict
        """
        from supervisely.project.project import Project

        temp_dir = tempfile.mkdtemp()

        data = Project.download_bin(
            self._api, self.project_info.id, batch_size=200, return_bytesio=True
        )
        data.seek(0)
        info = tarfile.TarInfo(name="version.bin")
        info.size = len(data.getvalue())
        chunk_size = 1024 * 1024 * 50  # 50 MiB
        tar_data = io.BytesIO()

        # Create a tarfile object that writes into the BytesIO object
        with tarfile.open(fileobj=tar_data, mode="w") as tar:
            tar.addfile(tarinfo=info, fileobj=data)
        data.close()
        # Reset the BytesIO object's cursor to the beginning
        tar_data.seek(0)
        zst_archive_path = os.path.join(temp_dir, "download.tar.zst")

        with open(zst_archive_path, "wb") as zst:
            while True:
                chunk = tar_data.read(chunk_size)
                if not chunk:
                    break
                zst.write(zstd.compress(chunk))
        file_info = self._api.file.upload(self.project_info.team_id, zst_archive_path, path)
        tar_data.close()
        remove_dir(temp_dir)
        return file_info
