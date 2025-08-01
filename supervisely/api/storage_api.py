# coding: utf-8
"""download/upload/manipulate files from/to Supervisely team files and cloud storages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import requests
from tqdm import tqdm
from typing_extensions import Literal

from supervisely.api.file_api import FileApi, FileInfo
from supervisely.api.module_api import ApiField
from supervisely.sly_logger import logger


class StorageApi(FileApi):
    """
    API for working with Files and Folders in Team Files and Cloud Storage. :class:`StorageApi<StorageApi>` object is immutable.

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
        files = api.storage.list(team_id, file_path)
    """

    def list2(self, team_id: int, path: str, recursive: bool = True) -> List[FileInfo]:
        """
        Method is not implemented.
        Use api.storage.list instead.
        Additionally, to get list of files in Team Files you can use api.file.list2.
        """
        raise NotImplementedError(
            "Method is not implemented. "
            "Use api.storage.list instead. "
            "Or use api.file.list2 to get list of files in Team Files."
        )

    def listdir(self, team_id: int, path: str, recursive: bool = False) -> List[str]:
        """
        Method is not implemented.
        Use api.storage.list instead.
        Additionally, to get list of files in Team Files you can use api.file.listdir.
        """
        raise NotImplementedError(
            "Method is not implemented. "
            "Use api.storage.list instead. "
            "Or use api.file.listdir to get list of files in Team Files."
        )

    def list(
        self,
        team_id: int,
        path: str,
        recursive: bool = True,
        return_type: Literal["dict", "fileinfo"] = "fileinfo",
        with_metadata: bool = True,
        include_files: bool = True,
        include_folders: bool = True,
        limit: Optional[int] = None,
    ) -> List[Union[Dict, FileInfo]]:
        """
        List of all or limited quantity files from the Team Files or Cloud Storages.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :param recursive: If True return all files recursively.
        :type recursive: bool
        :param return_type: The specified value between 'dict' or 'fileinfo'. By default: 'fileinfo'.
        :type return_type: str
        :param with_metadata: If True return files with metadata.
        :type with_metadata: bool
        :param include_files: If True return files infos.
        :type include_files: bool
        :param include_folders: If True return folders infos.
        :type include_folders: bool
        :param limit: Limit the number of files returned.
        :type limit: int
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

            # Option 1. Get information about files from Team Files:
            file_path = "/tmp/test_img/"
            files = api.storage.list(team_id, file_path)
            file = files[0]
            print(file.name)
            # Output: berries.jpeg

            print(files)
            # Output: [
            #   FileInfo(team_id=8, id=7660, user_id=7, name='00135.json', hash='z7Wv1a7WI...
            #   FileInfo(team_id=8, id=7661, user_id=7, name='01587.json', hash='La9+XtF2+...
            # ]


            # Option 2. Get information from Cloud Storages (S3, Azure, Google Cloud Storage)
            file_path = "s3://remote-img-test/test_img/"
            # or
            file_path = "azure://supervisely-test/test_img/"
            # or
            file_path = "google://sly-dev-test/test_img/"

            files = api.storage.list(team_id, file_path, return_type="dict", limit=2)
            print(files)
            # Output: [
            #     {
            #         "type": "file",
            #         "name": "berries-01.jpeg",
            #         "prefix": "test_img",
            #         "path": "azure://supervisely-test/test_img/berries-01.jpeg",
            #         "updatedAt": "2024-03-08T16:03:41.000Z",
            #         "meta": {
            #             "size": 3529930
            #         }
            #     },
            #     {
            #         "type": "file",
            #         "name": "berries-02.jpeg",
            #         "prefix": "test_img",
            #         "path": "azure://supervisely-test/test_img/berries-02.jpeg",
            #         "updatedAt": "2024-03-08T16:03:38.000Z",
            #         "meta": {
            #             "size": 1311144
            #         }
            #     }
            # ]
        """
        if not path.endswith("/"):
            path += "/"
        method = "file-storage.v2.list"
        json_body = {
            ApiField.TEAM_ID: team_id,
            ApiField.PATH: path,
            ApiField.RECURSIVE: recursive,
            ApiField.WITH_METADATA: with_metadata,
            ApiField.FILES: include_files,
            ApiField.FOLDERS: include_folders,
        }
        if limit is not None:
            json_body[ApiField.LIMIT] = limit

        try:
            first_response = self._api.post(method, json_body).json()
            data = first_response.get("entities", [])

            continuation_token = first_response.get("continuationToken", None)
            limit_exceeded = False
            if limit is not None and len(data) >= limit:
                limit_exceeded = True

            if continuation_token is None or limit_exceeded:
                pass
            else:
                while continuation_token is not None:
                    json_body["continuationToken"] = continuation_token
                    temp_resp = self._api.post(method, json_body)
                    temp_data = temp_resp.json().get("entities", [])
                    data.extend(temp_data)
                    continuation_token = temp_resp.json().get("continuationToken", None)
                    if limit is not None and len(data) >= limit:
                        limit_exceeded = True
                        break

        except requests.exceptions.RequestException as e:
            if self.is_on_agent(path) is True:
                logger.warn(f"Failed to list files on agent {path}: {repr(e)}", exc_info=True)
                return []
            else:
                raise e
        if limit_exceeded:
            data = data[:limit]

        if return_type == "fileinfo":
            results = []
            for info in data:
                info[ApiField.IS_DIR] = info[ApiField.TYPE] == "folder"
                results.append(self._convert_json_info(info))
            return results

        return data

    def _exists(self, team_id: int, remote_path: str, path_type: str) -> bool:
        """Checks if file or directory exists in Team Files / Cloud Storage / Agent."""

        parent_dir = os.path.dirname(remote_path.rstrip("/"))
        path_infos = self.list(team_id, parent_dir, recursive=False, return_type="dict")
        for info in path_infos:
            if info["type"] == path_type:
                if info["path"].rstrip("/") == remote_path.rstrip("/"):
                    return True
        return False

    def exists(self, team_id: int, remote_path: str) -> bool:
        """
        Checks if file exists in Team Files / Cloud Storage / Agent.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to File in Team Files.
        :type remote_path: str
        :return: True if file exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.storage.exists(8, "/999_App_Test/ds1/02163.json") # True
           file = api.storage.exists(8, "/999_App_Test/ds1/01587.json") # False
        """
        return self._exists(team_id, remote_path, "file")

    def dir_exists(self, team_id: int, remote_directory: str) -> bool:
        """
        Checks if directory exists in Team Files / Cloud Storage / Agent.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Remote path to directory in Team Files.
        :type remote_path: str
        :return: True if directory exists, otherwise False
        :rtype: :class:`bool`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           file = api.storage.dir_exists(8, "/999_App_Test/")   # True
           file = api.storage.dir_exists(8, "/10000_App_Test/") # False
        """
        return self._exists(team_id, remote_directory, "folder")

    def get_info_by_path(self, team_id: int, remote_path: str) -> FileInfo:
        """
        Gets File information by path in Team Files / Cloud Storage / Agent.

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
            file_info = api.storage.get_info_by_id(8, file_path)
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
        dir_path = os.path.dirname(remote_path.rstrip("/"))
        path_infos = self.list(team_id, dir_path, recursive=False)
        for info in path_infos:
            if info.path.rstrip("/") == remote_path.rstrip("/"):
                return info
        return None

    def get_info_by_id(self, id: int) -> FileInfo:
        """This method available only for Team Files."""
        return super().get_info_by_id(id)

    def remove_dir(self, team_id: int, remote_path: str) -> None:
        if not remote_path.endswith("/"):
            remote_path += "/"
        super().remove_dir(team_id, remote_path)

    def get_url(self, file_id: int) -> str:
        raise NotImplementedError()
