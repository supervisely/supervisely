from typing import Optional, List, Dict
import supervisely as sly
import supervisely.io.fs as fs
import supervisely.io.env as env
from supervisely.task.progress import Progress
from supervisely._utils import is_production
from supervisely.app.content import get_data_dir
import os
from supervisely.collection.key_indexed_collection import KeyIndexedCollection, KeyObject
from enum import Enum
from supervisely.sly_logger import logger


class FileContextType(Enum):
    FILE_FROM_TEAM_FILES = 0
    FOLDER_FROM_TEAM_FILES = 1
    FILE_FROM_URL = 2


class FilesContextItem(KeyObject):
    def __init__(self, key: str, type: FileContextType, link: str, location: str):
        self._key = key
        self._type = type
        self._link = link
        self._location = location

    def key(self) -> str:
        return self._key

    @property
    def type(self) -> FileContextType:
        return self._type

    @property
    def link(self) -> str:
        return self._link

    @property
    def location(self) -> str:
        return self._location


class FilesContext(KeyIndexedCollection):
    item_type = FilesContextItem

    def __init__(self, api: sly.Api, location: Dict[str, str] = None):
        self._api = api
        self._weights_file = None
        self._additional_files = None
        self._folder = None
        self._file_from_url = None
        items = self._prepare_model_files(location)
        super(FilesContext, self).__init__(items=items)

    def _download_from_location(self, location_key: str, location_link: str, local_files_path: str):
        if fs.is_on_agent(location_link) or is_production():
            team_id = env.team_id()
            basename = os.path.basename(os.path.normpath(location_link))
            local_path = os.path.join(local_files_path, basename)
            progress = Progress(f"Downloading {basename}...", 1, is_size=True, need_info_log=True)
            if fs.dir_exists(location_link) or fs.file_exists(location_link):
                # only during debug, has no effect in production
                local_path = os.path.abspath(location_link)
                if fs.dir_exists(local_path):
                    location_type = FileContextType.FOLDER_FROM_TEAM_FILES
                elif fs.file_exists(local_path):
                    location_type = FileContextType.FILE_FROM_TEAM_FILES
            elif self._api.file.dir_exists(team_id, location_link) and location_link.endswith(
                "/"
            ):  # folder from Team Files
                logger.info(f"Remote directory in Team Files: {location_link}")
                logger.info(f"Local directory: {local_path}")
                sizeb = self._api.file.get_directory_size(team_id, location_link)
                progress.set(current=0, total=sizeb)
                self._api.file.download_directory(
                    team_id,
                    location_link,
                    local_path,
                    progress.iters_done_report,
                )
                location_type = FileContextType.FOLDER_FROM_TEAM_FILES
                print(f"📥 Directory {basename} has been successfully downloaded from Team Files")
            elif self._api.file.exists(team_id, location_link):  # file from Team Files
                file_info = self._api.file.get_info_by_path(env.team_id(), location_link)
                progress.set(current=0, total=file_info.sizeb)
                self._api.file.download(
                    team_id, location_link, local_path, progress_cb=progress.iters_done_report
                )
                location_type = FileContextType.FILE_FROM_TEAM_FILES
                print(f"📥 File {basename} has been successfully downloaded from Team Files")
            else:  # external url
                fs.download(location_link, local_path, progress=progress)
                location_type = FileContextType.FILE_FROM_URL
                print(f"📥 File {basename} has been successfully downloaded.")
            print(f"File {basename} path: {local_path}")
        else:
            local_path = os.path.abspath(location_link)
            if fs.dir_exists(local_path):
                location_type = FileContextType.FOLDER_FROM_TEAM_FILES
            elif fs.file_exists(local_path):
                location_type = FileContextType.FILE_FROM_TEAM_FILES
        return FilesContextItem(
            location_key,
            location_type,
            location_link,
            local_path,
        )

    def _prepare_model_files(self, location: Dict[str, str] = None):
        if location is None:
            return None
        local_files_path = os.path.join(get_data_dir(), "model")
        fs.mkdir(local_files_path)
        items = []
        for k, loc in location.items():
            items.append(self._download_from_location(k, loc, local_files_path))
        return items
