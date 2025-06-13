# coding: utf-8
"""download/upload/manipulate files from/to Supervisely team files"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from time import time
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import aiofiles
import httpx
import requests
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
from typing_extensions import Literal

import supervisely.io.env as env
import supervisely.io.fs as sly_fs
from supervisely._utils import batched, rand_str, run_coroutine
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.api.remote_storage_api import RemoteStorageApi
from supervisely.io.fs import (
    ensure_base_path,
    get_file_ext,
    get_file_hash,
    get_file_hash_async,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
    get_or_create_event_loop,
    list_files_recursively,
    list_files_recursively_async,
    silent_remove,
)
from supervisely.io.fs_cache import FileCache
from supervisely.io.json import load_json_file
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress, tqdm_sly


class FileInfo(NamedTuple):
    """ """

    team_id: int
    id: int
    user_id: int
    name: str
    hash: str
    path: str
    storage_path: str
    mime: str
    ext: str
    sizeb: int
    created_at: str
    updated_at: str
    full_storage_url: str
    is_dir: bool


class FileApi(ModuleApiBase):
    """
    API for working with Files. :class:`FileApi<FileApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        team_id = 8
        file_path = "/999_App_Test/"
        files = api.file.list(team_id, file_path)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple FileInfo information about File.

        :Example:

         .. code-block:: python

            FileInfo(team_id=8,
                     id=7660,
                     user_id=7,
                     name='00135.json',
                     hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
                     path='/999_App_Test/ds1/00135.json',
                     storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
                     mime='application/json',
                     ext='json',
                     sizeb=261,
                     created_at='2021-01-11T09:04:17.959Z',
                     updated_at='2021-01-11T09:04:17.959Z',
                     full_storage_url='http://supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        return [
            ApiField.TEAM_ID,
            ApiField.ID,
            ApiField.USER_ID,
            ApiField.NAME,
            ApiField.HASH,
            ApiField.PATH,
            ApiField.STORAGE_PATH,
            ApiField.MIME2,
            ApiField.EXT2,
            ApiField.SIZEB2,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.FULL_STORAGE_URL,
            ApiField.IS_DIR,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **FileInfo**.
        """
        return "FileInfo"

    def list_on_agent(
        self,
        team_id: int,
        path: str,
        recursive: bool = True,
        return_type: Literal["dict", "fileinfo"] = "dict",
    ) -> List[Union[Dict, FileInfo]]:
        if self.is_on_agent(path) is False:
            raise ValueError(f"Data is not on agent: {path}")

        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(path)
        dirs_queue: List[str] = [path_in_agent_folder]

        results = []
        while len(dirs_queue) > 0:
            cur_dir = dirs_queue.pop(0)
            if cur_dir.endswith("/") is False:
                cur_dir += "/"
            response = self._api.post(
                "agents.storage.list",
                {ApiField.ID: agent_id, ApiField.TEAM_ID: team_id, ApiField.PATH: cur_dir},
            )
            items = response.json()
            for item in items:
                if item["type"] == "file":
                    results.append(item)
                elif item["type"] == "directory" and recursive is True:
                    dirs_queue.append(os.path.join(cur_dir, item["name"]))

        if return_type == "dict":
            return results
        elif return_type == "fileinfo":
            return [self._convert_json_info(info_json) for info_json in results]
        else:
            raise ValueError(
                "The specified value for the 'return_type' parameter should be either 'dict' or 'fileinfo'."
            )

    def list(
        self,
        team_id: int,
        path: str,
        recursive: bool = True,
        return_type: Literal["dict", "fileinfo"] = "dict",
    ) -> List[Union[Dict, FileInfo]]:
        """
        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :param recursive: If True return all files recursively.
        :type recursive: bool
        :param return_type: The specified value between 'dict' or 'fileinfo'. By default: 'dict'.
        :type return_type: str
        :return: List of all Files with information. See classes info_sequence and FileInfo
        :rtype: class List[Union[Dict, FileInfo]]
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
                load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Pass values into the API constructor (optional, not recommended)
            # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

            team_id = 8
            file_path = "/999_App_Test/"

            # Get information about file in dict way..
            files = api.file.list(team_id, file_path)
            file = files[0]
            print(file['id'])
            # Output: 7660

            print(files)
            # Output: [
            #     {
            #         "id":7660,
            #         "userId":7,
            #         "path":"/999_App_Test/ds1/00135.json",
            #         "storagePath":"/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json",
            #         "meta":{
            #             "ext":"json",
            #             "mime":"application/json",
            #             "size":261
            #         },
            #         "createdAt":"2021-01-11T09:04:17.959Z",
            #         "updatedAt":"2021-01-11T09:04:17.959Z",
            #         "hash":"z7Wv1a7WIC5HIJrfX/69XXrqtDaLxucSprWHoCxyq0M=",
            #         "fullStorageUrl":"http://supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json",
            #         "teamId":8,
            #         "name":"00135.json"
            #     },
            #     {
            #         "id":7661,
            #         "userId":7,
            #         "path":"/999_App_Test/ds1/01587.json",
            #         "storagePath":"/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/9/k/Hs/...json",
            #         "meta":{
            #             "ext":"json",
            #             "mime":"application/json",
            #             "size":252
            #         },
            #         "createdAt":"2021-01-11T09:04:18.099Z",
            #         "updatedAt":"2021-01-11T09:04:18.099Z",
            #         "hash":"La9+XtF2+cTlAqUE/I72e/xS12LqyH1+z<3T+SgD4CTU=",
            #         "fullStorageUrl":"http://supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/9/k/Hs/...json",
            #         "teamId":8,
            #         "name":"01587.json"
            #     }
            # ]

            # ..or as FileInfo with attributes:
            files = api.file.list(team_id, file_path, return_type='fileinfo')
            file = files[0]
            print(file.id)
            # Output: 7660

            print(files)
            # Output: [
            # FileInfo(team_id=8, id=7660, user_id=7, name='00135.json', hash='z7Wv1a7WI...
            # FileInfo(team_id=8, id=7661, user_id=7, name='01587.json', hash='La9+XtF2+...
            # ]
        """

        if not path.endswith("/") and recursive is False:
            path += "/"
        if self.is_on_agent(path) is True:
            return self.list_on_agent(team_id, path, recursive, return_type)

        response = self._api.post(
            "file-storage.list",
            {ApiField.TEAM_ID: team_id, ApiField.PATH: path, ApiField.RECURSIVE: recursive},
        )

        if return_type == "dict":
            return response.json()
        elif return_type == "fileinfo":
            return [self._convert_json_info(info_json) for info_json in response.json()]
        else:
            raise ValueError(
                "The specified value for the 'return_type' parameter should be either 'dict' or 'fileinfo'."
            )

    def list2(self, team_id: int, path: str, recursive: bool = True) -> List[FileInfo]:
        """
        Disclaimer: Method is not recommended. Use api.file.list instead

        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :param recursive: If True return all FileInfos recursively.
        :type recursive: bool
        :return: List of all Files with information. See class info_sequence
        :rtype: class List[FileInfo]
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
                load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Pass values into the API constructor (optional, not recommended)
            # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

            team_id = 9
            file_path = "/My_App_Test/"
            files = api.file.list2(team_id, file_path)

            print(files)
            # Output: [
            # FileInfo(team_id=9, id=18421, user_id=8, name='5071_3734_mot_video_002.tar.gz', hash='+0nrNoDjBxxJA...
            # FileInfo(team_id=9, id=18456, user_id=8, name='5164_4218_mot_video_bitmap.tar.gz', hash='fwtVI+iptY...
            # FileInfo(team_id=9, id=18453, user_id=8, name='all_vars.tar', hash='TVkUE+K1bnEb9QrdEm9akmHm/QEWPJK...
            # ]
        """
        return self.list(team_id=team_id, path=path, recursive=recursive, return_type="fileinfo")

    def listdir(self, team_id: int, path: str, recursive: bool = False) -> List[str]:
        """
        List dirs and files in the directiry with given path.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to directory.
        :type path: str
        :param recursive: If True return all paths recursively.
        :type recursive: bool
        :return: List of paths
        :rtype: :class:`List[str]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 8
            path = "/999_App_Test/"
            files = api.file.listdir(team_id, path)

            print(files)
            # Output: ["/999_App_Test/ds1", "/999_App_Test/image.png"]
        """
        files = self.list(team_id, path, recursive)
        files_paths = [file["path"] for file in files]
        return files_paths

    def get_directory_size(self, team_id: int, path: str) -> int:
        """
        Get directory size in the Team Files.
        If directory is on local agent, then optimized method will be used (without api requests)

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to Directory.
        :type path: str
        :return: Directory size in the Team Files
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 9
            path = "/My_App_Test/"
            size = api.file.get_directory_size(team_id, path)

            print(size)
            # Output: 3478687
        """
        if self.is_on_agent(path) is True:
            agent_id, path_in_agent_folder = self.parse_agent_id_and_path(path)
            if (
                agent_id == env.agent_id(raise_not_found=False)
                and env.agent_storage(raise_not_found=False) is not None
            ):
                path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
                logger.info(f"Optimized getting directory size from agent: {path_on_agent}")
                dir_size = sly_fs.get_directory_size(path_on_agent)
                return dir_size

        dir_size = 0
        file_infos = self.list(team_id=team_id, path=path, recursive=True, return_type="fileinfo")
        for file_info in file_infos:
            if file_info.sizeb is not None:
                dir_size += file_info.sizeb
        return dir_size

    def _download(
        self,
        team_id,
        remote_path,
        local_save_path,
        progress_cb=None,
        log_progress: bool = False,
    ):
        response = self._api.post(
            "file-storage.download",
            {ApiField.TEAM_ID: team_id, ApiField.PATH: remote_path},
            stream=True,
        )
        if progress_cb is not None:
            log_progress = False

        if log_progress and progress_cb is None:
            total_size = int(response.headers.get("Content-Length", 0))
            progress_cb = tqdm_sly(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading file",
                leave=True,
            )
        # print(response.headers)
        # print(response.headers['Content-Length'])
        ensure_base_path(local_save_path)
        with open(local_save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        cache: Optional[FileCache] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download File from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to File in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param cache: optional
        :type cache: FileCache, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_file = "/999_App_Test/ds1/01587.json"
            local_save_path = "/home/admin/Downloads/01587.json"

            api.file.download(8, path_to_file, local_save_path)
        """

        if self.is_on_agent(remote_path):
            self.download_from_agent(remote_path, local_save_path, progress_cb)
            return

        if cache is None:
            self._download(team_id, remote_path, local_save_path, progress_cb)
        else:
            file_info = self.get_info_by_path(team_id, remote_path)
            if file_info.hash is None:
                self._download(team_id, remote_path, local_save_path, progress_cb)
            else:
                cache_path = cache.check_storage_object(file_info.hash, get_file_ext(remote_path))
                if cache_path is None:
                    # file not in cache
                    self._download(team_id, remote_path, local_save_path, progress_cb)
                    if file_info.hash != get_file_hash(local_save_path):
                        raise KeyError(
                            f"Remote and local hashes are different (team id: {team_id}, file: {remote_path})"
                        )
                    cache.write_object(local_save_path, file_info.hash)
                else:
                    cache.read_object(file_info.hash, local_save_path)
                    if progress_cb is not None:
                        progress_cb(get_file_size(local_save_path))

    def is_on_agent(self, remote_path: str):
        return sly_fs.is_on_agent(remote_path)

    def parse_agent_id_and_path(self, remote_path: str) -> int:
        return sly_fs.parse_agent_id_and_path(remote_path)

    # TODO replace with download_async
    def download_from_agent(
        self,
        remote_path: str,
        local_save_path: str,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(remote_path)
        if (
            agent_id == env.agent_id(raise_not_found=False)
            and env.agent_storage(raise_not_found=False) is not None
        ):
            path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
            logger.info(f"Optimized download from agent: {path_on_agent}")
            sly_fs.copy_file(path_on_agent, local_save_path)
            if progress_cb is not None:
                progress_cb(sly_fs.get_file_size(path_on_agent))
            return

        response = self._api.post(
            "agents.storage.download",
            {ApiField.ID: agent_id, ApiField.PATH: path_in_agent_folder},
            stream=True,
        )
        ensure_base_path(local_save_path)
        with open(local_save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    # TODO replace with download_directory_async
    def download_directory(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Download Directory from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to Directory in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_save_path = "/home/admin/Downloads/My_local_test"

            api.file.download_directory(9, path_to_dir, local_save_path)
        """
        if not remote_path.endswith("/"):
            remote_path += "/"

        if self.is_on_agent(remote_path) is True:
            agent_id, path_in_agent_folder = self.parse_agent_id_and_path(remote_path)
            if (
                agent_id == env.agent_id(raise_not_found=False)
                and env.agent_storage(raise_not_found=False) is not None
            ):
                dir_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
                logger.info(f"Optimized download from agent: {dir_on_agent}")
                sly_fs.copy_dir_recursively(dir_on_agent, local_save_path)
                if progress_cb is not None:
                    progress_cb(sly_fs.get_directory_size(dir_on_agent))
                return

        local_temp_archive = os.path.join(local_save_path, "temp.tar")
        self.download(team_id, remote_path, local_temp_archive, cache=None, progress_cb=progress_cb)
        tr = tarfile.open(local_temp_archive)
        tr.extractall(local_save_path)
        silent_remove(local_temp_archive)
        temp_dir = os.path.join(local_save_path, rand_str(10))
        to_move_dir = os.path.join(local_save_path, os.path.basename(os.path.normpath(remote_path)))
        os.rename(to_move_dir, temp_dir)
        file_names = os.listdir(temp_dir)
        for file_name in file_names:
            shutil.move(os.path.join(temp_dir, file_name), local_save_path)
        shutil.rmtree(temp_dir)

    def download_input(
        self,
        save_path: str,
        unpack_if_archive: Optional[bool] = True,
        remove_archive: Optional[bool] = True,
        force: Optional[bool] = False,
        log_progress: bool = False,
    ) -> None:
        """Downloads data for application from input using environment variables.
        Automatically detects if data is a file or a directory and saves it to the specified directory.
        If data is an archive, it will be unpacked to the specified directory if unpack_if_archive is True.

        :param save_path: path to a directory where data will be saved
        :type save_path: str
        :param unpack_if_archive: if True, archive will be unpacked to the specified directory
        :type unpack_if_archive: Optional[bool]
        :param remove_archive: if True, archive will be removed after unpacking
        :type remove_archive: Optional[bool]
        :param force: if True, data will be downloaded even if it already exists in the specified directory
        :type force: Optional[bool]
        :param log_progress: if True, progress bar will be displayed
        :type log_progress: bool
        :raises RuntimeError:
            - if both file and folder paths not found in environment variables \n
            - if both file and folder paths found in environment variables (debug)
            - if team id not found in environment variables

        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Application is started...
            save_path = "/my_app_data"
            api.file.download_input(save_path)

            # The data is downloaded to the specified directory.
        """
        remote_file_path = env.file(raise_not_found=False)
        remote_folder_path = env.folder(raise_not_found=False)
        team_id = env.team_id()

        sly_fs.mkdir(save_path)

        if remote_file_path is None and remote_folder_path is None:
            raise RuntimeError(
                "Both file and folder paths not found in environment variables. "
                "Please, specify one of them."
            )
        elif remote_file_path is not None and remote_folder_path is not None:
            raise RuntimeError(
                "Both file and folder paths found in environment variables. "
                "Please, specify only one of them."
            )
        if team_id is None:
            raise RuntimeError("Team id not found in environment variables.")

        if remote_file_path is not None:
            file_name = sly_fs.get_file_name_with_ext(remote_file_path)
            local_file_path = os.path.join(save_path, file_name)

            if os.path.isfile(local_file_path) and not force:
                logger.info(
                    f"The file {local_file_path} already exists. "
                    "Download is skipped, if you want to download it again, "
                    "use force=True."
                )
                return

            sly_fs.silent_remove(local_file_path)

            progress_cb = None
            file_info = self.get_info_by_path(team_id, remote_file_path)
            if log_progress is True and file_info is not None:
                progress = Progress(
                    f"Downloading {remote_file_path}", file_info.sizeb, is_size=True
                )
                progress_cb = progress.iters_done_report
            if self.is_on_agent(remote_file_path):
                self.download_from_agent(remote_file_path, local_file_path, progress_cb=progress_cb)
            else:
                self.download(team_id, remote_file_path, local_file_path, progress_cb=progress_cb)
            if unpack_if_archive and sly_fs.is_archive(local_file_path):
                sly_fs.unpack_archive(local_file_path, save_path)
                if remove_archive:
                    sly_fs.silent_remove(local_file_path)
                else:
                    logger.info(
                        f"Achive {local_file_path} was unpacked, but not removed. "
                        "If you want to remove it, use remove_archive=True."
                    )
        elif remote_folder_path is not None:
            folder_name = os.path.basename(os.path.normpath(remote_folder_path))
            local_folder_path = os.path.join(save_path, folder_name)
            if os.path.isdir(local_folder_path) and not force:
                logger.info(
                    f"The folder {folder_name} already exists. "
                    "Download is skipped, if you want to download it again, "
                    "use force=True."
                )
                return

            sly_fs.remove_dir(local_folder_path)

            progress_cb = None
            if log_progress is True:
                sizeb = self.get_directory_size(team_id, remote_folder_path)
                progress = Progress(f"Downloading: {remote_folder_path}", sizeb, is_size=True)
                progress_cb = progress.iters_done_report
            self.download_directory(
                team_id, remote_folder_path, local_folder_path, progress_cb=progress_cb
            )

    def _upload_legacy(self, team_id, src, dst):
        """ """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        item = get_file_name_with_ext(dst)
        content_dict = {}
        content_dict[ApiField.NAME] = item

        dst_dir = os.path.dirname(dst)
        if not dst_dir.endswith("/"):
            dst_dir += "/"
        content_dict[ApiField.PATH] = dst_dir  # os.path.basedir ...
        content_dict["file"] = (
            item,
            path_to_bytes_stream(src),
            mimetypes.MimeTypes().guess_type(src)[0],
        )
        encoder = MultipartEncoder(fields=content_dict)
        resp = self._api.post("file-storage.upload?teamId={}".format(team_id), encoder)
        return resp.json()

    def upload(
        self, team_id: int, src: str, dst: str, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> FileInfo:
        """
        Upload File to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file path.
        :type src: str
        :param dst: Path to File in Team Files.
        :type dst: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_path = "/home/admin/Downloads/01587.json"
            dst_remote_path = "/999_App_Test/ds1/01587.json"

            api.file.upload(8, src_path, dst_remote_path)
        """
        return self.upload_bulk(team_id, [src], [dst], progress_cb)[0]

    def upload_bulk(
        self,
        team_id: int,
        src_paths: List[str],
        dst_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[FileInfo]:
        """
        Upload Files to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file paths.
        :type src: List[str]
        :param dst: Destination paths for Files to Team Files.
        :type dst: List[str]
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about Files. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[FileInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_paths = ["/home/admin/Downloads/01587.json", "/home/admin/Downloads/01588.json","/home/admin/Downloads/01589.json"]
            dst_remote_paths = ["/999_App_Test/ds1/01587.json", "/999_App_Test/ds1/01588.json", "/999_App_Test/ds1/01589.json"]

            api.file.upload_bulk(8, src_paths, dst_remote_paths)
        """

        def _group_files_generator(
            src_paths: List[str], dst_paths: List[str], limit: int = 20 * 1024 * 1024
        ):
            if limit is None:
                return src_paths, dst_paths
            group_src = []
            group_dst = []
            total_size = 0
            for src, dst in zip(src_paths, dst_paths):
                size = os.path.getsize(src)
                if total_size > 0 and total_size + size > limit:
                    yield group_src, group_dst
                    group_src = []
                    group_dst = []
                    total_size = 0
                group_src.append(src)
                group_dst.append(dst)
                total_size += size
            if total_size > 0:
                yield group_src, group_dst

        file_infos = []
        for src, dst in _group_files_generator(src_paths, dst_paths):
            file_infos.extend(self._upload_bulk(team_id, src, dst, progress_cb))
        return file_infos

    def _upload_bulk(
        self,
        team_id: int,
        src_paths: List[str],
        dst_paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[FileInfo]:

        def path_to_bytes_stream(path):
            return open(path, "rb")

        content_dict = []
        for idx, (src, dst) in enumerate(zip(src_paths, dst_paths)):
            name = get_file_name_with_ext(dst)
            content_dict.append((ApiField.NAME, name))
            dst_dir = os.path.dirname(dst)
            if not dst_dir.endswith("/"):
                dst_dir += "/"
            content_dict.append((ApiField.PATH, dst_dir))
            content_dict.append(
                (
                    "file",
                    (
                        name,
                        path_to_bytes_stream(src),
                        mimetypes.MimeTypes().guess_type(src)[0],
                    ),
                )
            )
        encoder = MultipartEncoder(fields=content_dict)

        # progress = None
        # if progress_logger is not None:
        #     progress = Progress("Uploading", encoder.len, progress_logger, is_size=True)
        #
        # def _print_progress(monitor):
        #     if progress is not None:
        #         progress.set_current_value(monitor.bytes_read)
        #         print(monitor.bytes_read, monitor.len)
        # last_percent = 0
        # def _progress_callback(monitor):
        #     cur_percent = int(monitor.bytes_read * 100.0 / monitor.len)
        #     if cur_percent - last_percent > 10 or cur_percent == 100:
        #         api.task.set_fields(task_id, [{"field": "data.previewProgress", "payload": cur_percent}])
        #     last_percent = cur_percent

        if progress_cb is None:
            data = encoder
        else:
            try:
                data = MultipartEncoderMonitor(encoder, progress_cb.get_partial())
            except AttributeError:
                data = MultipartEncoderMonitor(encoder, progress_cb)
        resp = self._api.post("file-storage.bulk.upload?teamId={}".format(team_id), data)
        results = [self._convert_json_info(info_json) for info_json in resp.json()]

        return results

    def rename(self, old_name: str, new_name: str) -> None:
        """
        Renames file in Team Files

        :param old_name: Old File name.
        :type old_name: str
        :param new_name: New File name.
        :type new_name: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # NotImplementedError('Method is not supported')
        """
        pass

    def remove_from_agent(self, team_id: int, path: str) -> None:
        raise NotImplementedError()
        agent_id, path_in_agent_folder = self.parse_agent_id_and_path(path)
        if (
            agent_id == env.agent_id(raise_not_found=False)
            and env.agent_storage(raise_not_found=False) is not None
        ):
            # path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
            # logger.info(f"Optimized download from agent: {path_on_agent}")
            # sly_fs.copy_file(path_on_agent, local_save_path)
            return

    def remove(self, team_id: int, path: str) -> None:
        """
        Removes a file from the Team Files. If the specified path is a directory,
        the entire directory (including all recursively included files) will be removed.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path in Team Files.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove(8, "/999_App_Test/ds1/01587.json") # remove file
            api.file.remove(8, "/999_App_Test/ds1/") # remove folder
        """

        if self.is_on_agent(path) is True:
            # self.remove_from_agent(team_id, path)
            logger.warn(
                f"Data '{path}' is on agent. Method does not support agent storage. Remove your data manually on the computer with agent."
            )
            return

        self._api.post("file-storage.remove", {ApiField.TEAM_ID: team_id, ApiField.PATH: path})

    def remove_file(self, team_id: int, path: str) -> None:
        """
        Removes file from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File in Team Files.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove_file(8, "/999_App_Test/ds1/01587.json")
        """

        file_info = self.get_info_by_path(team_id, path)

        if file_info is None:
            raise ValueError(
                f"File not found in Team files. Maybe you entered directory? (Path: '{path}')"
            )

        self.remove(team_id, path)

    def remove_dir(self, team_id: int, path: str, silent: bool = False) -> None:
        """
        Removes folder from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to folder in Team Files.
        :type path: str
        :param silent: Ignore if directory not exists.
        :type silent: bool

        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove_dir(8, "/999_App_Test/ds1/")
        """

        if not path.endswith("/"):
            raise ValueError("Please add a slash in the end to recognize path as a directory.")

        if silent is False:
            if not self.dir_exists(team_id, path):
                raise ValueError(f"Folder not found in Team files. (Path: '{path}')")

        self.remove(team_id, path)

    def remove_batch(
        self,
        team_id: int,
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        batch_size: int = 1000,
    ) -> None:
        """
        Removes list of files from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param paths: List of paths to Files in Team Files.
        :type paths: List[str]
        :param progress_cb: Function for tracking progress.
        :type progress_cb: tqdm or callable, optional
        :param batch_size: Number of files to remove in one request. Default is 1000. Maximum is 20000.
        :type batch_size: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            paths_to_del = [
                "/999_App_Test/ds1/01587.json",
                "/999_App_Test/ds1/01588.json",
                "/999_App_Test/ds1/01587.json"
            ]
            api.file.remove_batch(8, paths_to_del)
        """
        if batch_size > 20000:
            logger.warning(
                "Batch size is more than maximum and automatically reduced to 20000. "
                "If you get an error, try reducing the batch size."
            )
            batch_size = 20000
        elif batch_size < 100:
            logger.warning("Batch size is less than minimum and automatically increased to 100.")
            batch_size = 100

        for paths_batch in batched(paths, batch_size=batch_size):
            for path in paths_batch:
                if self.is_on_agent(path) is True:
                    logger.warning(
                        f"Data '{path}' is on agent. File skipped. Method does not support agent storage. Remove your data manually on the computer with agent."
                    )
                    paths_batch.remove(path)

            self._api.post(
                "file-storage.bulk.remove", {ApiField.TEAM_ID: team_id, ApiField.PATHS: paths_batch}
            )
            if progress_cb is not None:
                progress_cb(len(paths_batch))

    def exists(self, team_id: int, remote_path: str, recursive: bool = True) -> bool:
        """
        Checks if file exists in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to File in Team Files.
        :type remote_path: str
        :param recursive: If True makes more checks and slower, if False makes less checks and faster.
        :type recursive: bool
        :return: True if file exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.exists(8, "/999_App_Test/ds1/02163.json") # True
           file = api.file.exists(8, "/999_App_Test/ds1/01587.json") # False
        """
        path_infos = self.list(team_id, remote_path, recursive)
        for info in path_infos:
            if info["path"] == remote_path:
                return True
        return False

    def dir_exists(self, team_id: int, remote_directory: str, recursive: bool = True) -> bool:
        """
        Checks if directory exists in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to directory in Team Files.
        :type remote_path: str
        :param recursive: If True makes more checks and slower, if False makes less checks and faster.
        :type recursive: bool
        :return: True if directory exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.dir_exists(8, "/999_App_Test/")   # True
           file = api.file.dir_exists(8, "/10000_App_Test/") # False
        """
        files_infos = self.list(team_id, remote_directory, recursive)
        if len(files_infos) > 0:
            return True
        return False

    def get_free_name(self, team_id: int, path: str) -> str:
        """
        Adds suffix to the end of the file name.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Remote path to file in Team Files.
        :type path: str
        :return: New File name with suffix at the end
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.file.get_free_name(8, "/999_App_Test/ds1/02163.json")
           print(file)
           # Output: /999_App_Test/ds1/02163_000.json
        """
        directory = Path(path).parent
        name = get_file_name(path)
        ext = get_file_ext(path)
        res_name = name
        suffix = 0

        def _combine(suffix: int = None):
            res = "{}/{}".format(directory, res_name)
            if suffix is not None:
                res += "_{:03d}".format(suffix)
            if ext:
                res += "{}".format(ext)
            return res

        res_path = _combine()
        while self.exists(team_id, res_path):
            res_path = _combine(suffix)
            suffix += 1
        return res_path

    def get_url(self, file_id: int) -> str:
        """
        Gets URL for the File by ID.

        :param file_id: File ID in Supervisely.
        :type file_id: int
        :return: File URL
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file_id = 7660
           file_url = sly.api.file.get_url(file_id)
           print(file_url)
           # Output: http://supervisely.com/files/7660
        """
        return f"/files/{file_id}"

    def get_info_by_path(self, team_id: int, remote_path: str) -> FileInfo:
        """
        Gets File information by path in Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to file in Team Files.
        :type remote_path: str
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file_path = "/999_App_Test/ds1/00135.json"
            file_info = api.file.get_info_by_id(8, file_path)
            print(file_info)
            # Output: FileInfo(team_id=8,
            #                  id=7660,
            #                  user_id=7,
            #                  name='00135.json',
            #                  hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
            #                  path='/999_App_Test/ds1/00135.json',
            #                  storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
            #                  mime='application/json',
            #                  ext='json',
            #                  sizeb=261,
            #                  created_at='2021-01-11T09:04:17.959Z',
            #                  updated_at='2021-01-11T09:04:17.959Z',
            #                  full_storage_url='http://supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        if self.is_on_agent(remote_path) is True:
            path_infos = self.list_on_agent(team_id, os.path.dirname(remote_path), recursive=False)
            for info in path_infos:
                if info["path"] == remote_path:
                    return self._convert_json_info(info)
        else:
            path_infos = self.list(team_id, remote_path)
            for info in path_infos:
                if info["path"] == remote_path:
                    return self._convert_json_info(info)
        return None

    def _convert_json_info(self, info: dict, skip_missing=True) -> FileInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        # if res.storage_path is not None:
        #    res = res._replace(full_storage_url=urllib.parse.urljoin(self._api.server_address, res.storage_path))
        return FileInfo(**res._asdict())

    def get_info_by_id(self, id: int) -> FileInfo:
        """
        Gets information about File by ID.

        :param id: File ID in Supervisely.
        :type id: int
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`FileInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file_id = 7660
            file_info = api.file.get_info_by_id(file_id)
            print(file_info)
            # Output: FileInfo(team_id=8,
            #                  id=7660,
            #                  user_id=7,
            #                  name='00135.json',
            #                  hash='z7Hv9a7WIC5HIJrfX/69KVrvtDaLqucSprWHoCxyq0M=',
            #                  path='/999_App_Test/ds1/00135.json',
            #                  storage_path='/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json',
            #                  mime='application/json',
            #                  ext='json',
            #                  sizeb=261,
            #                  created_at='2021-01-11T09:04:17.959Z',
            #                  updated_at='2021-01-11T09:04:17.959Z',
            #                  full_storage_url='http://supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
        """
        resp = self._api.post("file-storage.info", {ApiField.ID: id})
        return self._convert_json_info(resp.json())

    def get_free_dir_name(self, team_id: int, dir_path: str) -> str:
        """
        Adds suffix to the end of the Directory name.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param dir_path: Path to Directory in Team Files.
        :type dir_path: str
        :return: New Directory name with suffix at the end
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           new_dir_name = api.file.get_free_dir_name(9, "/My_App_Test")
           print(new_dir_name)
           # Output: /My_App_Test_001
        """
        res_dir = dir_path.rstrip("/")
        if not self.dir_exists(team_id, res_dir + "/"):
            return res_dir

        low, high = 0, 1
        while self.dir_exists(team_id, f"{res_dir}_{high:03d}/"):
            low = high
            high *= 2

        while low < high:
            mid = (low + high) // 2
            if self.dir_exists(team_id, f"{res_dir}_{mid:03d}/"):
                low = mid + 1
            else:
                high = mid

        return f"{res_dir}_{low:03d}"

    def upload_directory(
        self,
        team_id: int,
        local_dir: str,
        remote_dir: str,
        change_name_if_conflict: Optional[bool] = True,
        progress_size_cb: Optional[Union[tqdm, Callable]] = None,
        replace_if_conflict: Optional[bool] = False,
    ) -> str:
        """
        Upload Directory to Team Files from local path.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param local_dir: Path to local Directory.
        :type local_dir: str
        :param remote_dir: Path to Directory in Team Files.
        :type remote_dir: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param progress_size_cb: Function for tracking download progress.
        :type progress_size_cb: Progress, optional
        :return: Path to Directory in Team Files
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_path = "/home/admin/Downloads/My_local_test"

            api.file.upload_directory(9, local_path, path_to_dir)
        """
        if not remote_dir.startswith("/"):
            if not RemoteStorageApi.is_bucket_url(remote_dir):
                remote_dir = "/" + remote_dir

        if self.dir_exists(team_id, remote_dir):
            if change_name_if_conflict is True:
                res_remote_dir = self.get_free_dir_name(team_id, remote_dir)
            elif replace_if_conflict is True:
                res_remote_dir = remote_dir
            else:
                raise FileExistsError(
                    f"Directory {remote_dir} already exists in your team (id={team_id})"
                )
        else:
            res_remote_dir = remote_dir

        local_files = list_files_recursively(local_dir)
        remote_files = []
        dir_parts = local_dir.strip("/").split("/")
        for file in local_files:
            path_parts = file.strip("/").split("/")
            path_parts = path_parts[len(dir_parts) :]
            remote_parts = [res_remote_dir.rstrip("/")] + path_parts
            remote_file = "/".join(remote_parts)
            remote_files.append(remote_file)

        for local_paths_batch, remote_files_batch in zip(
            batched(local_files, batch_size=50), batched(remote_files, batch_size=50)
        ):
            self.upload_bulk(team_id, local_paths_batch, remote_files_batch, progress_size_cb)
        return res_remote_dir

    def load_dotenv_from_teamfiles(
        self, remote_path: str = None, team_id: int = None, override: int = False
    ) -> None:
        """Loads .env file from Team Files into environment variables.
        If remote_path or team_id is not specified, it will be taken from environment variables.

        :param remote_path: Path to .env file in Team Files.
        :type remote_path: str, optional
        :param team_id: Team ID in Supervisely.
        :type team_id: int, optional
        :param override: If True, existing environment variables will be overridden.
        :type override: bool, optional

        :Usage example:

            .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            api.file.load_dotenv_from_teamfiles()
            # All variables from .env file are loaded into environment variables.
        """
        # If remote_path or team_id is not specified, it will be taken from environment variables.
        remote_path = remote_path or env.file(raise_not_found=False)
        team_id = team_id or env.team_id(raise_not_found=False)

        if not remote_path or not team_id or not remote_path.endswith(".env"):
            return

        try:
            file_name = sly_fs.get_file_name(remote_path)

            # Use timestamp to avoid conflicts with existing files.
            timestamp = int(time())
            local_save_path = os.path.join(os.getcwd(), f"{file_name}_{timestamp}.env")

            # Download .env file from Team Files.
            self.download(team_id=team_id, remote_path=remote_path, local_save_path=local_save_path)

            # Load .env file into environment variables and then remove it.
            load_dotenv(local_save_path, override=override)
            sly_fs.silent_remove(local_save_path)

            logger.debug(f"Loaded .env file from team files: {remote_path}")
        except Exception as e:
            logger.debug(f"Failed to load .env file from team files: {remote_path}. Error: {e}")

    def get_json_file_content(self, team_id: int, remote_path: str, download: bool = False) -> dict:
        """
        Get JSON file content.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to JSON file in Team Files.
        :type remote_path: str
        :param download: If True, download file in temp dir to get content.
        :type download: bool, optional
        :return: JSON file content
        :rtype: :class:`dict` or :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api("https://app.supervisely.com", "YourAccessToken")

            file_content = api.file.get_json_file_content()
            print(file_content)
        """

        MB = 1024 * 1024
        max_readable_size = 100 * MB

        file_info = self._api.file.get_info_by_path(team_id, remote_path)
        if file_info:
            if file_info.mime != "application/json":
                raise ValueError(f"File is not JSON: {remote_path}")
            content = None
            if file_info.sizeb <= max_readable_size or not download:
                response = requests.get(file_info.full_storage_url)
                if response.status_code != 200:
                    download = True
                else:
                    content = response.json()
            if file_info.sizeb > max_readable_size or download:
                temp_path = os.path.join(tempfile.mkdtemp(), "temp.json")
                self._download(team_id, remote_path, temp_path)
                content = load_json_file(temp_path)
                sly_fs.remove_dir(temp_path)
            return content
        else:
            raise FileNotFoundError(f"File not found in Team Files at path: {remote_path}")

    async def _download_async(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: dict = None,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
    ):
        """
        Download file from Team Files or connected Cloud Storage.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to File in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param range_start: Start byte position for partial download.
        :type range_start: int, optional
        :param range_end: End byte position for partial download.
        :type range_end: int, optional
        :param headers: Additional headers for request.
        :type headers: dict, optional
        :param check_hash: If True, checks hash of downloaded file.
                        Check is not supported for partial downloads.
                        When range is set, hash check is disabled.
        :type check_hash: bool
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        """
        api_method = "file-storage.download"

        if range_start is not None or range_end is not None:
            check_hash = False
            headers = headers or {}
            headers["Range"] = f"bytes={range_start or ''}-{range_end or ''}"
            logger.debug(f"File: {remote_path}. Setting Range header: {headers['Range']}")

        json_body = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: remote_path,
            **self._api.additional_fields,
        }

        writing_method = "ab" if range_start not in [0, None] else "wb"

        ensure_base_path(local_save_path)
        hash_to_check = None
        async with aiofiles.open(local_save_path, writing_method) as fd:
            async for chunk, hhash in self._api.stream_async(
                method=api_method,
                method_type="POST",
                data=json_body,
                headers=headers,
                range_start=range_start,
                range_end=range_end,
            ):
                await fd.write(chunk)
                hash_to_check = hhash
                if progress_cb is not None and progress_cb_type == "size":
                    progress_cb(len(chunk))
            await fd.flush()

        if check_hash:
            if hash_to_check is not None:
                downloaded_file_hash = await get_file_hash_async(local_save_path)
                if hash_to_check != downloaded_file_hash:
                    raise RuntimeError(
                        f"Downloaded hash of file path: '{remote_path}' does not match the expected hash: {downloaded_file_hash} != {hash_to_check}"
                    )
        if progress_cb is not None and progress_cb_type == "number":
            progress_cb(1)

    async def download_async(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        cache: Optional[FileCache] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
    ) -> None:
        """
        Download File from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to File in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore
        :param cache: Cache object for storing files.
        :type cache: FileCache, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_file = "/999_App_Test/ds1/01587.json"
            local_save_path = "/path/to/save/999_App_Test/ds1/01587.json"
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(api.file.download_async(8, path_to_file, local_save_path))
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        async with semaphore:
            if self.is_on_agent(remote_path):
                # for optimized download from agent
                # in other agent cases download will be performed as usual
                agent_id, path_in_agent_folder = self.parse_agent_id_and_path(remote_path)
                if (
                    agent_id == env.agent_id(raise_not_found=False)
                    and env.agent_storage(raise_not_found=False) is not None
                ):
                    path_on_agent = os.path.normpath(env.agent_storage() + path_in_agent_folder)
                    logger.info(f"Optimized download from agent: {path_on_agent}")
                    await sly_fs.copy_file_async(
                        path_on_agent, local_save_path, progress_cb, progress_cb_type
                    )
                    return

            if cache is None:
                await self._download_async(
                    team_id,
                    remote_path,
                    local_save_path,
                    progress_cb=progress_cb,
                    progress_cb_type=progress_cb_type,
                )
            else:
                file_info = self.get_info_by_path(team_id, remote_path)
                if file_info.hash is None:
                    await self._download_async(
                        team_id,
                        remote_path,
                        local_save_path,
                        progress_cb=progress_cb,
                        progress_cb_type=progress_cb_type,
                    )
                else:
                    cache_path = cache.check_storage_object(
                        file_info.hash, get_file_ext(remote_path)
                    )
                    if cache_path is None:
                        # file not in cache
                        await self._download_async(
                            team_id,
                            remote_path,
                            local_save_path,
                            progress_cb=progress_cb,
                            progress_cb_type=progress_cb_type,
                        )
                        if file_info.hash != await get_file_hash_async(local_save_path):
                            raise KeyError(
                                f"Remote and local hashes are different (team id: {team_id}, file: {remote_path})"
                            )
                        await cache.write_object_async(local_save_path, file_info.hash)
                    else:
                        await cache.read_object_async(file_info.hash, local_save_path)
                        if progress_cb is not None and progress_cb_type == "size":
                            progress_cb(get_file_size(local_save_path))
                        if progress_cb is not None and progress_cb_type == "number":
                            progress_cb(1)

    async def download_bulk_async(
        self,
        team_id: int,
        remote_paths: List[str],
        local_save_paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        caches: Optional[List[FileCache]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
    ):
        """
        Download multiple Files from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_paths: List of paths to Files in Team Files.
        :type remote_paths: List[str]
        :param local_save_paths: List of local save paths.
        :type local_save_paths: List[str]
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore
        :param caches: List of cache objects for storing files.
        :type caches: List[FileCache], optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            paths_to_files = [
                "/999_App_Test/ds1/01587.json",
                "/999_App_Test/ds1/01588.json",
                "/999_App_Test/ds1/01587.json"
            ]
            local_paths = [
                "/path/to/save/999_App_Test/ds1/01587.json",
                "/path/to/save/999_App_Test/ds1/01588.json",
                "/path/to/save/999_App_Test/ds1/01587.json"
            ]
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.file.download_bulk_async(8, paths_to_files, local_paths)
                )
        """
        if len(remote_paths) == 0:
            return

        if len(remote_paths) != len(local_save_paths):
            raise ValueError(
                f"Length of remote_paths and local_save_paths must be equal: {len(remote_paths)} != {len(local_save_paths)}"
            )
        elif caches is not None and len(remote_paths) != len(caches):
            raise ValueError(
                f"Length of remote_paths and caches must be equal: {len(remote_paths)} != {len(caches)}"
            )

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        tasks = []
        for remote_path, local_path, cache in zip(
            remote_paths, local_save_paths, caches or [None] * len(remote_paths)
        ):
            task = self.download_async(
                team_id,
                remote_path,
                local_path,
                semaphore=semaphore,
                cache=cache,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_directory_async(
        self,
        team_id: int,
        remote_path: str,
        local_save_path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Download Directory from Team Files to local path asynchronously.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to Directory in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads.
        :type semaphore: asyncio.Semaphore
        :param show_progress: If True show download progress.
        :type show_progress: bool
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import asyncio

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/files/folder"
            local_path = "path/to/local/folder"

            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.file.download_directory_async(9, path_to_dir, local_path)
                )
        """

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        if not remote_path.endswith("/"):
            remote_path += "/"

        tasks = []
        files = self._api.storage.list(  # to avoid method duplication in storage api
            team_id,
            remote_path,
            recursive=True,
            include_folders=False,
            with_metadata=False,
        )
        sizeb = sum([file.sizeb for file in files])
        if show_progress:
            progress_cb = tqdm_sly(
                total=sizeb, desc=f"Downloading files from directory", unit="B", unit_scale=True
            )
        else:
            progress_cb = None

        for file in files:
            task = self.download_async(
                team_id,
                file.path,
                os.path.join(local_save_path, file.path[len(remote_path) :]),
                semaphore=semaphore,
                progress_cb=progress_cb,
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_input_async(
        self,
        save_path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        unpack_if_archive: Optional[bool] = True,
        remove_archive: Optional[bool] = True,
        force: Optional[bool] = False,
        show_progress: bool = False,
    ) -> None:
        """Asynchronously downloads data for the application, using a path from file/folder selector.
        The application adds this path to environment variables, which the method then reads.
        Automatically detects if data is a file or a directory and saves it to the specified directory.
        If data is an archive, it will be unpacked to the specified directory if unpack_if_archive is True.

        :param save_path: path to a directory where data will be saved
        :type save_path: str
        :param semaphore: Semaphore for limiting the number of simultaneous downloads
        :type semaphore: asyncio.Semaphore
        :param unpack_if_archive: if True, archive will be unpacked to the specified directory
        :type unpack_if_archive: Optional[bool]
        :param remove_archive: if True, archive will be removed after unpacking
        :type remove_archive: Optional[bool]
        :param force: if True, data will be downloaded even if it already exists in the specified directory
        :type force: Optional[bool]
        :param show_progress: if True, progress bar will be displayed
        :type show_progress: bool
        :raises RuntimeError:
            - if both file and folder paths not found in environment variables \n
            - if both file and folder paths found in environment variables (debug)
            - if team id not found in environment variables

        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly
            import asyncio

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            # Application is started...
            save_path = "/my_app_data"
            loop = sly.utils.get_or_create_event_loop()
            loop.run_until_complete(
                    api.file.download_input_async(save_path)
                )

            # The data is downloaded to the specified directory.
        """

        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        remote_file_path = env.file(raise_not_found=False)
        remote_folder_path = env.folder(raise_not_found=False)
        team_id = env.team_id()

        sly_fs.mkdir(save_path)

        if remote_file_path is None and remote_folder_path is None:
            raise RuntimeError(
                "Both file and folder paths not found in environment variables. "
                "Please, specify one of them."
            )
        elif remote_file_path is not None and remote_folder_path is not None:
            raise RuntimeError(
                "Both file and folder paths found in environment variables. "
                "Please, specify only one of them."
            )
        if team_id is None:
            raise RuntimeError("Team id not found in environment variables.")

        if remote_file_path is not None:
            file_name = sly_fs.get_file_name_with_ext(remote_file_path)
            local_file_path = os.path.join(save_path, file_name)

            if os.path.isfile(local_file_path) and not force:
                logger.info(
                    f"The file {local_file_path} already exists. "
                    "Download is skipped, if you want to download it again, "
                    "use force=True."
                )
                return

            sly_fs.silent_remove(local_file_path)

            progress_cb = None
            file_info = self.get_info_by_path(team_id, remote_file_path)
            if show_progress is True and file_info is not None:
                progress_cb = tqdm_sly(
                    desc=f"Downloading {remote_file_path}",
                    total=file_info.sizeb,
                    unit="B",
                    unit_scale=True,
                )
            await self.download_async(
                team_id,
                remote_file_path,
                local_file_path,
                semaphore=semaphore,
                progress_cb=progress_cb,
            )
            if unpack_if_archive and sly_fs.is_archive(local_file_path):
                await sly_fs.unpack_archive_async(local_file_path, save_path)
                if remove_archive:
                    sly_fs.silent_remove(local_file_path)
                else:
                    logger.info(
                        f"Achive {local_file_path} was unpacked, but not removed. "
                        "If you want to remove it, use remove_archive=True."
                    )
        elif remote_folder_path is not None:
            folder_name = os.path.basename(os.path.normpath(remote_folder_path))
            local_folder_path = os.path.join(save_path, folder_name)
            if os.path.isdir(local_folder_path) and not force:
                logger.info(
                    f"The folder {folder_name} already exists. "
                    "Download is skipped, if you want to download it again, "
                    "use force=True."
                )
                return

            sly_fs.remove_dir(local_folder_path)

            await self.download_directory_async(
                team_id,
                remote_folder_path,
                local_folder_path,
                semaphore=semaphore,
                show_progress=show_progress,
            )

    async def upload_async(
        self,
        team_id: int,
        src: str,
        dst: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        # chunk_size: int = 1024 * 1024, #TODO add with resumaple api
        # check_hash: bool = True, #TODO add with resumaple api
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
    ) -> None:
        """
        Upload file from local path to Team Files asynchronously.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local path to file.
        :type src: str
        :param dst: Path to save file in Team Files.
        :type dst: str
        :param semaphore: Semaphore for limiting the number of simultaneous uploads.
        :type semaphore: asyncio.Semaphore, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

            .. code-block:: python

                import supervisely as sly
                import asyncio

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                path_to_file = "/path/to/local/file/01587.json"
                path_to_save = "/files/01587.json"
                loop = sly.utils.get_or_create_event_loop()
                loop.run_until_complete(
                    api.file.upload_async(8, path_to_file, path_to_save)
                )
        """
        api_method = "file-storage.upload"
        headers = {"Content-Type": "application/octet-stream"}
        # sha256 = await get_file_hash_async(src) #TODO add with resumaple api
        json_body = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: dst,
            # "sha256": sha256, #TODO add with resumaple api
        }
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()
        logger.debug(f"Uploading with async to: {dst}. Semaphore: {semaphore}")
        async with semaphore:
            async with aiofiles.open(src, "rb") as fd:

                async def file_chunk_generator():
                    while True:
                        chunk = await fd.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        if progress_cb is not None and progress_cb_type == "size":
                            progress_cb(len(chunk))
                        yield chunk

                async for chunk, _ in self._api.stream_async(
                    method=api_method,
                    method_type="POST",
                    data=file_chunk_generator(),  # added as required, but not used inside
                    headers=headers,
                    content=file_chunk_generator(),  # used instead of data inside stream_async
                    params=json_body,
                ):
                    pass
                if progress_cb is not None and progress_cb_type == "number":
                    progress_cb(1)

    async def upload_bulk_async(
        self,
        team_id: int,
        src_paths: List[str],
        dst_paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        # chunk_size: int = 1024 * 1024, #TODO add with resumaple api
        # check_hash: bool = True, #TODO add with resumaple api
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
        enable_fallback: Optional[bool] = True,
    ) -> None:
        """
        Upload multiple files from local paths to Team Files asynchronously.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src_paths: List of local paths to files.
        :type src_paths: List[str]
        :param dst_paths: List of paths to save files in Team Files.
        :type dst_paths: List[str]
        :param semaphore: Semaphore for limiting the number of simultaneous uploads.
        :type semaphore: asyncio.Semaphore, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :param enable_fallback: If True, the method will fallback to synchronous upload if an error occurs.
        :type enable_fallback: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

            .. code-block:: python

                import supervisely as sly
                import asyncio

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                paths_to_files = [
                    "/path/to/local/file/01587.json",
                    "/path/to/local/file/01588.json",
                    "/path/to/local/file/01589.json"
                ]
                paths_to_save = [
                    "/files/01587.json",
                    "/files/01588.json",
                    "/files/01589.json"
                ]
                loop = sly.utils.get_or_create_event_loop()
                loop.run_until_complete(
                    api.file.upload_bulk_async(8, paths_to_files, paths_to_save)
                )
        """
        try:
            if semaphore is None:
                semaphore = self._api.get_default_semaphore()
            tasks = []
            for src, dst in zip(src_paths, dst_paths):
                task = asyncio.create_task(
                    self.upload_async(
                        team_id=team_id,
                        src=src,
                        dst=dst,
                        semaphore=semaphore,
                        # chunk_size=chunk_size, #TODO add with resumaple api
                        # check_hash=check_hash, #TODO add with resumaple api
                        progress_cb=progress_cb,
                        progress_cb_type=progress_cb_type,
                    )
                )
                tasks.append(task)
            for task in tasks:
                await task
        except Exception as e:
            if enable_fallback:
                logger.warning(
                    f"Upload files bulk asynchronously failed. Fallback to synchronous upload.",
                    exc_info=True,
                )
                if progress_cb is not None and progress_cb_type == "number":
                    logger.warning(
                        "Progress callback type 'number' is not supported for synchronous upload. "
                        "Progress callback will be disabled."
                    )
                    progress_cb = None
                self.upload_bulk(
                    team_id=team_id,
                    src_paths=src_paths,
                    dst_paths=dst_paths,
                    progress_cb=progress_cb,
                )
            else:
                raise e

    async def upload_directory_async(
        self,
        team_id: int,
        local_dir: str,
        remote_dir: str,
        change_name_if_conflict: Optional[bool] = True,
        progress_size_cb: Optional[Union[tqdm, Callable]] = None,
        replace_if_conflict: Optional[bool] = False,
        enable_fallback: Optional[bool] = True,
    ) -> str:
        """
        Upload Directory to Team Files from local path.
        Files are uploaded asynchronously.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param local_dir: Path to local Directory.
        :type local_dir: str
        :param remote_dir: Path to Directory in Team Files.
        :type remote_dir: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param progress_size_cb: Function for tracking download progress.
        :type progress_size_cb: Progress, optional
        :param replace_if_conflict: If True, replace existing dir.
        :type replace_if_conflict: bool, optional
        :param enable_fallback: If True, the method will fallback to synchronous upload if an error occurs.
        :type enable_fallback: bool, optional
        :return: Path to Directory in Team Files
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_path = "/home/admin/Downloads/My_local_test"

            api.file.upload_directory(9, local_path, path_to_dir)
        """
        try:
            if not remote_dir.startswith("/"):
                remote_dir = "/" + remote_dir

            if self.dir_exists(team_id, remote_dir):
                if change_name_if_conflict is True:
                    res_remote_dir = self.get_free_dir_name(team_id, remote_dir)
                elif replace_if_conflict is True:
                    res_remote_dir = remote_dir
                else:
                    raise FileExistsError(
                        f"Directory {remote_dir} already exists in your team (id={team_id})"
                    )
            else:
                res_remote_dir = remote_dir

            local_files = await list_files_recursively_async(local_dir)
            dir_prefix = local_dir.rstrip("/") + "/"
            remote_files = [
                res_remote_dir.rstrip("/") + "/" + file[len(dir_prefix) :] for file in local_files
            ]

            await self.upload_bulk_async(
                team_id=team_id,
                src_paths=local_files,
                dst_paths=remote_files,
                progress_cb=progress_size_cb,
            )
        except Exception as e:
            if enable_fallback:
                logger.warning(
                    f"Upload directory asynchronously failed. Fallback to synchronous upload.",
                    exc_info=True,
                )
                res_remote_dir = self.upload_directory(
                    team_id=team_id,
                    local_dir=local_dir,
                    remote_dir=res_remote_dir,
                    change_name_if_conflict=change_name_if_conflict,
                    progress_size_cb=progress_size_cb,
                    replace_if_conflict=replace_if_conflict,
                )
            else:
                raise e
        return res_remote_dir

    def upload_directory_fast(
        self,
        team_id: int,
        local_dir: str,
        remote_dir: str,
        change_name_if_conflict: Optional[bool] = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        replace_if_conflict: Optional[bool] = False,
        enable_fallback: Optional[bool] = True,
    ) -> str:
        """
        Upload Directory to Team Files from local path in fast mode.
        Files are uploaded asynchronously. If an error occurs, the method will fallback to synchronous upload.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param local_dir: Path to local Directory.
        :type local_dir: str
        :param remote_dir: Path to Directory in Team Files.
        :type remote_dir: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param progress_cb: Function for tracking download progress in bytes.
        :type progress_cb: Progress, optional
        :param replace_if_conflict: If True, replace existing dir.
        :type replace_if_conflict: bool, optional
        :param enable_fallback: If True, the method will fallback to synchronous upload if an error occurs.
        :type enable_fallback: bool, optional
        :return: Path to Directory in Team Files
        :rtype: :class:`str`
        """
        coroutine = self.upload_directory_async(
            team_id=team_id,
            local_dir=local_dir,
            remote_dir=remote_dir,
            change_name_if_conflict=change_name_if_conflict,
            progress_size_cb=progress_cb,
            replace_if_conflict=replace_if_conflict,
            enable_fallback=enable_fallback,
        )
        return run_coroutine(coroutine)

    def upload_bulk_fast(
        self,
        team_id: int,
        src_paths: List[str],
        dst_paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "size",
        enable_fallback: Optional[bool] = True,
    ) -> None:
        """
        Upload multiple files from local paths to Team Files in fast mode.
        Files are uploaded asynchronously. If an error occurs, the method will fallback to synchronous upload.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src_paths: List of local paths to files.
        :type src_paths: List[str]
        :param dst_paths: List of paths to save files in Team Files.
        :type dst_paths: List[str]
        :param semaphore: Semaphore for limiting the number of simultaneous uploads.
        :type semaphore: asyncio.Semaphore, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
                                "size" is used to track the number of transferred bytes.
                                "number" is used to track the number of transferred files.
        :type progress_cb_type: Literal["number", "size"], optional
        :param enable_fallback: If True, the method will fallback to synchronous upload if an error occurs.
        :type enable_fallback: bool, optional
        :return: None
        :rtype: :class:`NoneType`
        """
        coroutine = self.upload_bulk_async(
            team_id=team_id,
            src_paths=src_paths,
            dst_paths=dst_paths,
            semaphore=semaphore,
            progress_cb=progress_cb,
            progress_cb_type=progress_cb_type,
            enable_fallback=enable_fallback,
        )
        return run_coroutine(coroutine)
