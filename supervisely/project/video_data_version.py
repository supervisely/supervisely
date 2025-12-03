from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, Union

from supervisely._utils import logger
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.api.project_api import ProjectInfo, ProjectType


class VideoVersionInfo(NamedTuple):
    """
    Object with parameters from Supervisely that describes the version of a video project.
    Mirrors `VersionInfo` from `data_version.py`, but kept separate to avoid
    coupling image and video formats.
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


class VideoDataVersion(ModuleApiBase):
    """
    Class for managing **video project** versions.
    """

    def __init__(self, api):
        """
        Class for managing versions of video projects.
        """
        from supervisely import Api

        self._api: Api = api
        self.__storage_dir: str = "/system/versions/"
        self.__version_format: str = "video_arrow_v1"
        self.project_info: Optional[ProjectInfo] = None
        self.project_dir: Optional[str] = None
        self.versions_path: Optional[str] = None
        self.versions: Optional[dict] = None

    @staticmethod
    def info_sequence():
        """
        NamedTuple `VideoVersionInfo` with API Fields containing information about Video Project Version.
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
        NamedTuple name - **VideoVersionInfo**.
        """
        return "VideoVersionInfo"

    def initialize(self, project_info: Union[ProjectInfo, int]):
        """
        Initialize video project versioning state for given project.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        """
        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)
        self.project_info: ProjectInfo = project_info
        self.project_dir = f"{self.__storage_dir}{self.project_info.id}"
        self.versions_path = f"{self.project_dir}/versions.json"
        self.versions = self.get_map(self.project_info, do_initialization=False)
        if self.project_info.version is None:
            self._create_warning_system_file()

    def get_list(self, project_id: int, filters: Optional[List] = None) -> List[VideoVersionInfo]:
        """
        Get list of video project versions.
        Thin wrapper over `projects.versions.list` with the same contract as in `DataVersion`.
        """
        data = {ApiField.PROJECT_ID: project_id}
        if filters:
            data[ApiField.FILTER] = filters
        return self.get_list_all_pages("projects.versions.list", data)

    def get_id_by_number(self, project_id: int, version_num: int) -> Optional[int]:
        """
        Get version ID by version number for a video project.
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

    def get_map(
        self, project_info: Union[ProjectInfo, int], do_initialization: bool = True
    ) -> dict:
        """
        Get video project versions map from storage (versions.json in Team Files).
        Mirrors the behavior of `DataVersion.get_map`, but with its own format identifier.
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
        Save video project versions map to storage.
        """
        import os
        import tempfile
        from supervisely.io import json as sly_json
        from supervisely.io.fs import remove_dir

        if initialize:
            self.initialize(project_info)

        temp_dir = tempfile.mkdtemp()
        local_versions = os.path.join(temp_dir, "versions.json")
        sly_json.dump_json_file(self.versions, local_versions)
        file_info = self._api.file.upload(
            self.project_info.team_id, local_versions, self.versions_path
        )
        if file_info is None:
            raise RuntimeError("Failed to save video versions map")

        remove_dir(temp_dir)

    def create(
        self,
        project_info: Union[ProjectInfo, int],
        version_title: Optional[str] = None,
        version_description: Optional[str] = None,
    ) -> Optional[int]:
        """
        Create a new version for a video project.
        """
        from supervisely.project.video_project import VideoProject

        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)

        if project_info.type != ProjectType.VIDEOS.value:
            raise ValueError(f"Project with id {project_info.id} is not a video project")

        if (
            "app.supervise.ly" in self._api.server_address
            or "app.supervisely.com" in self._api.server_address
        ):
            if self._api.team.get_info_by_id(project_info.team_id).usage.plan == "free":
                logger.warning(
                    "Project versioning is not available for teams with Free plan. "
                    "Please upgrade to Pro to enable versioning."
                )
                return None

        self.initialize(project_info)
        path = self._generate_save_path()
        latest = self._get_latest_id()

        try:
            version_id, commit_token = self.reserve(project_info.id)
            # @TODO: remove log
            logger.debug(f"version_id: {version_id}, commit_token: {commit_token}")
        except Exception as e:
            logger.error(f"Failed to reserve video version. Exception: {e}")
            return None

        if version_id is None and commit_token is None:
            return latest

        try:
            snapshot_io = VideoProject.download_bin(
                self._api,
                project_info.id,
                dest_dir=None,
                dataset_ids=None,
                batch_size=50,
                log_progress=False,
                progress_cb=None,
                return_bytesio=True,
            )

            import os
            import tempfile
            from supervisely.io.fs import remove_dir

            temp_dir = tempfile.mkdtemp()
            local_archive = os.path.join(temp_dir, "snapshot.tar.zst")
            with open(local_archive, "wb") as f:
                f.write(snapshot_io.read())

            file_info = self._api.file.upload(
                self.project_info.team_id,
                local_archive,
                path,
            )
            if file_info is None:
                raise RuntimeError("Failed to upload video version snapshot to Team Files")
            remove_dir(temp_dir)

            latest_number = (
                int(self.versions[str(latest)]["number"])
                if (latest and str(latest) in self.versions)
                else 0
            )
            self.versions[version_id] = {
                "path": path,
                "updated_at": project_info.updated_at,
                "previous": latest,
                "number": latest_number + 1,
                "schema": self.__version_format,
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
                logger.error(
                    f"Video version creation failed. Reservation was cancelled. Exception: {e}"
                )
            else:
                logger.error(
                    "Failed to cancel video version reservation when handling exception. "
                    "You can cancel your reservation on the web under the Versions tab of the project. "
                    f"Exception: {e}"
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
        Commit video project version (same endpoint as for image versions).
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
            raise RuntimeError("Failed to commit video project version")

    def reserve(self, project_id: int, retries: int = 6) -> Tuple[int, str]:
        """
        Reserve video project version.
        Logic is identical to `DataVersion.reserve`, but kept separate to avoid
        accidental behavior changes.
        """
        import time
        import requests

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
                details = e.response.json().get("details", {})
                if details.get("useExistingVersion"):
                    version = details.get("version", {})
                    version_id = version.get("id")
                    version_name = version.get("version")
                    logger.info(
                        f"No changes to the video project since the last version "
                        f"'{version_name}' with ID '{version_id}'"
                    )
                    return (None, None)
                elif "is already committing" in details.get("message", ""):
                    if retry_delay >= max_delay:
                        raise RuntimeError(
                            "Failed to reserve video version. Another process is already committing "
                            "a version. Maximum number of attempts reached."
                        )
                    version = details.get("version", {}).get("version")
                    logger.info(
                        f"Video project is already committing version '{version}'. "
                        f"Retrying in {retry_delay} seconds."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2

    def cancel_reservation(self, version_id: int, commit_token: str) -> bool:
        """
        Cancel version reservation for a video project.
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
    ) -> Optional[ProjectInfo]:
        """
        Restore a video project to a specific version.
        Version can be specified by ID or number.

        The restored project will be uploaded as a **new** project, similar to how
        `Project.upload_bin` behaves for image projects, but using
        `VideoProject.upload_bin`.
        """
        from supervisely.project.video_project import VideoProject

        if version_id is None and version_num is None:
            raise ValueError("Either version_id or version_num must be provided")

        if isinstance(project_info, int):
            project_info = self._api.project.get_info_by_id(project_info)

        if project_info.type != ProjectType.VIDEOS.value:
            raise ValueError(f"Project with id {project_info.id} is not a video project")

        self.initialize(project_info)

        if version_num is not None:
            resolved_id = None
            for key, value in self.versions.items():
                if not isinstance(value, dict):
                    continue
                if value.get("number") == version_num:
                    resolved_id = key
                    break
            if resolved_id is None:
                raise ValueError(f"Version {version_num} does not exist for this video project")
            version_id = int(resolved_id)
        else:
            if str(version_id) not in self.versions:
                raise ValueError(f"Version {version_id} does not exist for this video project")

        vinfo = self.versions[str(version_id)]
        backup_path = vinfo.get("path")
        schema = vinfo.get("schema", self.__version_format)

        if schema != self.__version_format:
            raise RuntimeError(
                f"Unsupported video version schema '{schema}' "
                f"(expected '{self.__version_format}')"
            )

        if backup_path is None:
            logger.warning(
                f"Video project can't be restored to version {vinfo.get('number')} "
                f"because it doesn't have restore point."
            )
            return None

        snapshot_io = self._download_snapshot(backup_path)
        new_project_info = VideoProject.upload_bin(
            self._api,
            snapshot_io,
            workspace_id=self.project_info.workspace_id,
            project_name=None,
            with_custom_data=True,
            log_progress=False,
            progress_cb=None,
            skip_missed=skip_missed_entities,
        )
        return new_project_info

    def _create_warning_system_file(self):
        """
        Create a file in the system directory to indicate that you cannot manually modify its contents.

        Path = /system/DO_NOT_DELETE_ANYTHING_HERE.txt
        """
        import tempfile

        warning_file = "/system/DO_NOT_DELETE_ANYTHING_HERE.txt"
        if not self._api.file.exists(self.project_info.team_id, warning_file, recursive=False):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open(temp_file.name, "w") as f:
                f.write("This directory is managed by Supervisely. Do not modify its contents.")
            self._api.file.upload(self.project_info.team_id, temp_file.name, warning_file)

    def _generate_save_path(self) -> str:
        """
        Generate a path for the new video version archive where it will be saved in the Team Files.
        Archive format: {timestamp}.tar.zst
        """
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = os.path.join(self.project_dir, timestamp + ".tar.zst")
        return path

    def _get_latest_id(self) -> Optional[int]:
        """
        Get the ID of the latest version from the versions map (versions.json).
        """
        if not self.versions:
            return None
        latest = self.versions.get("latest", None)
        if not latest:
            return None
        return latest

    def _download_snapshot(self, path: str) -> _io.BytesIO:
        """
        Download stored snapshot (.tar.zst) for a video project version into memory.
        """
        import io
        import os
        import tempfile
        from supervisely.io.fs import remove_dir

        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "download.tar.zst")
        try:
            self._api.file.download(self.project_info.team_id, path, local_path)
            with open(local_path, "rb") as f:
                data = f.read()
            return io.BytesIO(data)
        except Exception as e:
            raise RuntimeError(f"Failed to download video version snapshot: {e}")
        finally:
            remove_dir(temp_dir)
