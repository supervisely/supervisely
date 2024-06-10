import os
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import List, NamedTuple, Tuple, Union

import requests
import zstd

from supervisely._utils import batched, logger
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.api.project_api import ProjectInfo
from supervisely.io import json
from supervisely.io.fs import archive_directory, remove_dir, silent_remove
from supervisely.project.project_meta import ProjectMeta


class VersionInfo(NamedTuple):
    """ """

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


class DataVersion(ModuleApiBase):

    def __init__(self, api):
        """
        Class for managing project versions.
        """
        from supervisely import Api

        self._api: Api = api
        self.project_info = None
        self.storage_dir = None
        self.project_dir = None
        self.versions_path = None
        self.versions = None

    @staticmethod
    def info_sequence():
        """
        NamedTuple VersionInfo with API Fields containing information about Project Version.

        :Example:

         .. code-block:: python

            VersionInfo(id=999
                        )
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
        self.storage_dir: str = "/system/versions/"  # directory in Team Files
        self.project_dir: str = os.path.join(
            self.storage_dir, str(self.project_info.id)
        )  # directory in Team Files
        self.versions_path: str = os.path.join(self.project_dir, "versions.json")
        self.versions: dict = self.load_json(self.project_info, initialize=False)
        self._create_warning_system_file()

    def get_list(self, project_id: int) -> List[VersionInfo]:
        """
        Get list of project versions.

        :param project_id: Project ID
        :type project_id: int
        :return: List of project versions
        :rtype: List[VersionInfo]
        """
        return self.get_list_all_pages(
            "projects.versions.list",
            {ApiField.PROJECT_ID: project_id},
        )

    def load_json(self, project_info: Union[ProjectInfo, int], initialize: bool = True):
        """
        Get project versions from storage.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param initialize: Initialize project versions. Set to False for internal use.
        :type initialize: bool
        :return: Project versions
        :rtype: dict
        """
        if initialize:
            self.initialize(project_info)
        file_info = self._api.file.get_info_by_path(self.project_info.team_id, self.versions_path)
        if file_info:
            response = requests.get(file_info.full_storage_url, stream=True)
            response.raise_for_status()
            versions = response.json()
        else:
            versions = {}
        return versions

    def upload_json(self, project_info: Union[ProjectInfo, int], initialize: bool = True):
        """
        Save project versions to storage.

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
        else:
            remove_dir(temp_dir)

    def create(self, project_info: Union[ProjectInfo, int]):
        """
        Create a new project version.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :return: Version ID
        :rtype: int
        """

        self.initialize(project_info)
        path = self._generate_save_path()
        latest = self._get_latest_id()
        version_id, commit_token = self.reserve(project_info.id)
        if version_id is None and commit_token is None:
            return latest
        try:
            current_state = self._get_current_state()
            previous_state = (
                self.versions[latest]["state"] if latest else {"datasets": {}, "items": {}}
            )
            changes = self._compute_changes(previous_state, current_state)
            file_info = self._upload_files(path, changes)
            self.versions[version_id] = {
                "path": path,
                "previous": latest,
                "changes": changes,
                "state": current_state,
            }
            self.versions["latest"] = version_id
            self.upload_json(project_info, initialize=False)
        except Exception as e:
            if self.cancel_reservation(version_id, commit_token):
                raise e
            else:
                raise RuntimeError(f"Failed to cancel reservation when handling exception: {e}")
        self.commit(
            version_id,
            commit_token,
            project_info.updated_at,
            file_info.id,
        )
        return version_id

    def commit(self, version_id: int, commit_token: str, updated_at: str, file_id: int):
        """
        Commit a project version.

        :param version_id: Version ID
        :type version_id: int
        :param commit_token: Commit token
        :type commit_token: str
        :param updated_at: Updated at timestamp
        :type updated_at: str
        :param file_id: File ID
        :type file_id: int
        :return: None
        """
        response = self._api.post(
            "projects.versions.commit",
            {
                ApiField.ID: version_id,
                ApiField.COMMIT_TOKEN: commit_token,
                ApiField.PROJECT_UPDATED_AT: updated_at,
                ApiField.TEAM_FILE_ID: file_id,
            },
        )
        commit_info = response.json()
        if not commit_info.get("success"):
            raise RuntimeError("Failed to commit version")

    def restore(self, project_info: Union[ProjectInfo, int], target_version: int):
        """
        Restore project to a specific version.

        :param project_info: ProjectInfo object or project ID
        :type project_info: Union[ProjectInfo, int]
        :param target_version: Version ID to restore to
        :type target_version: int
        :return: None
        """

        self.initialize(project_info)

        if target_version not in self.versions:
            raise ValueError(f"Version {target_version} does not exist")

        # Start from the latest version and walk back to the target version
        version_chain = []
        current_version = self._get_latest_id()

        while current_version != target_version:
            version_chain.append(current_version)
            current_version = self.versions[current_version]["previous"]

        # Apply changes in reverse order
        for version_id in reversed(version_chain):
            self._apply_changes(self.versions[version_id]["changes"], reverse=True)

    def reserve(self, project_id: int) -> Tuple[int, str]:
        """
        Reserve a project for versioning.

        :param project_id: Project ID
        :type project_id: int
        :return: Version ID and commit token
        :rtype: Tuple[int, str]
        """
        try:
            response = self._api.post(
                "projects.versions.reserve", {ApiField.PROJECT_ID: project_id}
            )
            reserve_info = response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.json().get("details", {}).get("version").get("useExistingVersion"):
                version_id = e.response.json().get("details", {}).get("version").get("id")
                version = e.response.json().get("details", {}).get("version").get("version")
                logger.info(
                    f"No changes to the project since the last version '{version}' with ID '{version_id}'"
                )
                return (None, None)
            else:
                raise e

        return reserve_info.get(ApiField.ID), reserve_info.get(ApiField.COMMIT_TOKEN)

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
            "projects.versions.reserve",
            {ApiField.ID: version_id, ApiField.COMMIT_TOKEN: commit_token},
        )
        reserve_info = response.json()
        return True if reserve_info.get("success") else False

    def _create_warning_system_file(self):
        """
        Create a file in the system directory to indicate that you cannot manually modify its contents.
        """
        warning_file = "/system/DO_NOT_DELETE_ANYTHING_HERE.txt"
        if not self._api.file.exists(self.project_info.team_id, warning_file, recursive=False):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open(temp_file.name, "w") as f:
                f.write("This directory is managed by Supervisely. Do not modify its contents.")
            self._api.file.upload(self.project_info.team_id, temp_file.name, warning_file)

    def _apply_changes(self, changes, reverse=False):
        for state_type in ["datasets", "items"]:
            deleted_datasets = set()
            items_to_delete = set()
            datasets_to_restore = set()
            items_to_restore = set()
            datasets_to_update = set()
            items_to_update = set()
            if reverse:
                for id, info in changes["added"][state_type].items():
                    if state_type == "datasets":
                        self._api.dataset.remove(id)
                        deleted_datasets.add(id)
                    elif state_type == "items":
                        if info["parent"] in deleted_datasets:
                            continue
                        else:
                            items_to_delete.add(id)
                for item in changes["deleted"][state_type].items():
                    if state_type == "datasets":
                        datasets_to_restore.add(item)
                    else:
                        items_to_restore.add(item)
            else:
                for item in changes["added"][state_type].items():
                    if state_type == "datasets":
                        datasets_to_restore.add(item)
                    else:
                        items_to_restore.add(item)
                for id in changes["deleted"][state_type].keys():
                    if state_type == "datasets":
                        self._api.dataset.remove(id)
                        deleted_datasets.add(id)
                    elif state_type == "items":
                        if info["parent"] in deleted_datasets:
                            continue
                        else:
                            items_to_delete.add(id)

            for item in changes["modified"][state_type].items():
                if state_type == "datasets":
                    datasets_to_update.add(item)
                else:
                    items_to_update.add(item)

            self._api.image.remove_batch(items_to_delete)
            self._restore_datasets(datasets_to_restore)
            self._restore_items(items_to_restore)

    def _compute_changes(self, old_state: dict, new_state: dict):
        changes = {
            "added": {"datasets": {}, "items": {}},
            "deleted": {"datasets": {}, "items": {}},
            "modified": {"datasets": {}, "items": {}},
        }

        for state_type in ["datasets", "items"]:
            added = {
                k: v for k, v in new_state[state_type].items() if k not in old_state[state_type]
            }
            deleted = {
                k: v for k, v in old_state[state_type].items() if k not in new_state[state_type]
            }

            if state_type == "datasets":
                modified = {
                    k: v
                    for k, v in new_state[state_type].items()
                    if k in old_state[state_type] and old_state[state_type][k]["name"] != v["name"]
                }
            else:  # state_type == "items"
                modified = {
                    k: v
                    for k, v in new_state[state_type].items()
                    if k in old_state[state_type]
                    and old_state[state_type][k]["updated_at"] != v["updated_at"]
                }

            changes["added"][state_type] = added
            changes["deleted"][state_type] = deleted
            changes["modified"][state_type] = modified

        return changes

    def _generate_save_path(self):
        """
        Create a path for the new version.

        :return: Path for the new version
        :rtype: str
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        path = os.path.join(self.project_dir, timestamp + ".tar.zst")
        return path

    def _get_latest_id(self):
        latest = self.versions.get("latest", None)
        if not latest:
            return None
        return latest

    def _get_current_state(self):
        """
        Scan project items and datasets to create a map of project state.

        :return: Project state map
        :rtype: dict
        """

        current_state = {"datasets": {}, "items": {}}

        for parents, dataset_info in self._api.dataset.tree(self.project_info.id):
            parent_id = parents[-1] if parents else 0
            current_state["datasets"][dataset_info.id] = {
                "name": dataset_info.name,
                "parent": parent_id,
            }

            for image_list in self._api.image.get_list_generator(dataset_info.id):
                for image in image_list:
                    current_state["items"][image.id] = {
                        "name": image.name,
                        "hash": image.hash,
                        "parent": dataset_info.id,
                        "updated_at": image.updated_at,
                    }
        return current_state

    def _restore_datasets(self, datasets) -> list:
        created_datasets = []
        for dataset in datasets:
            _, info = dataset
            dataset_info = self._api.dataset.create(
                self.project_info.id, info["name"], parent_id=info["parent"]
            )
            if dataset_info is None:
                raise RuntimeError(f"Failed to restore dataset {info['name']}")
            created_datasets.append(dataset_info)

        return created_datasets

    def _restore_items(self, items):
        items_by_parent = defaultdict(lambda: {"names": [], "hashes": []})

        for item in items:
            _, info = item
            items_by_parent[info["parent"]]["names"].append(info["name"])
            items_by_parent[info["parent"]]["hashes"].append(info["hash"])
        for parent_id, items in items_by_parent.items():
            self._api.image.upload_hashes(parent_id, items["names"], items["hashes"])

        return

    def _upload_files(self, path: str, changes: dict):
        """
        Save annotation files for items in project that were added or modified in the current version to the repository.
        Structure of the repository follows Supervisely format.

        :param changes: Changes between current and previous version
        :type changes: dict
        :return: File info
        :rtype: dict
        """
        from supervisely.project.project import (
            Dataset,
            OpenMode,
            Project,
            _maybe_append_image_extension,
        )

        temp_dir = tempfile.mkdtemp()

        items_to_save = list({**changes["added"]["items"], **changes["modified"]["items"]}.keys())
        filters = [{"field": "id", "operator": "in", "value": items_to_save}]

        project_fs = Project(temp_dir, OpenMode.CREATE)
        meta = ProjectMeta.from_json(
            self._api.project.get_meta(self.project_info.id, with_settings=True)
        )
        project_fs.set_meta(meta)

        for parents, dataset_info in self._api.dataset.tree(self.project_info.id):
            dataset_path = Dataset._get_dataset_path(dataset_info.name, parents)
            dataset_name = dataset_info.name
            dataset_id = dataset_info.id

            dataset = project_fs.create_dataset(dataset_name, dataset_path)

            images_to_download = self._api.image.get_list(dataset_id, filters)
            ann_info_list = self._api.annotation.download_batch(dataset_id, items_to_save)
            img_name_to_ann = {ann.image_id: ann.annotation for ann in ann_info_list}
            for img_info_batch in batched(images_to_download):
                images_nps = [None] * len(img_info_batch)
                for index, _ in enumerate(images_nps):
                    img_info = img_info_batch[index]
                    image_name = _maybe_append_image_extension(img_info.name, img_info.ext)

                    dataset.add_item_np(
                        item_name=image_name,
                        img=None,
                        ann=img_name_to_ann[img_info.id],
                        img_info=None,
                    )

        archive_path = os.path.join(os.path.dirname(temp_dir), "download.tar")
        archive_directory(temp_dir, archive_path)
        zst_archive_path = archive_path + ".zst"
        with open(archive_path, "rb") as tar:
            with open(zst_archive_path, "wb") as zst:
                zst.write(zstd.ZSTD_compress(tar.read()))
        file_info = self._api.file.upload(self.project_info.team_id, zst_archive_path, path)
        silent_remove(archive_path)
        silent_remove(zst_archive_path)
        remove_dir(temp_dir)
        return file_info
