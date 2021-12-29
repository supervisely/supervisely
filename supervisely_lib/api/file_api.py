# coding: utf-8

from __future__ import annotations

from typing import NamedTuple, List, Dict, Optional

import os
import shutil
import tarfile
from pathlib import Path
import urllib
from supervisely_lib.api.module_api import ModuleApiBase, ApiField
from supervisely_lib.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import mimetypes
from supervisely_lib.task.progress import Progress
from supervisely_lib.io.fs_cache import FileCache
from supervisely_lib.io.fs import get_file_hash, get_file_ext, get_file_size, list_files_recursively, silent_remove, get_file_name

class FileApi(ModuleApiBase):
    """
    API for working with Files. :class:`FileApi<FileApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

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
                     full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
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
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **FileInfo**.
        """
        return 'FileInfo'

    def list(self, team_id: int, path: str) -> List[Dict]:
        """
        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :return: List of all Files with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 8
            file_path = "/999_App_Test/"
            files = api.file.list(team_id, file_path)

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
            #         "fullStorageUrl":"http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json",
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
            #         "fullStorageUrl":"http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/9/k/Hs/...json",
            #         "teamId":8,
            #         "name":"01587.json"
            #     }
            # ]
        """
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        return response.json()

    def list2(self, team_id: int, path: str) -> List[NamedTuple]:
        """
        List of files in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to File or Directory.
        :type path: str
        :return: List of all Files with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

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
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        results = [self._convert_json_info(info_json) for info_json in response.json()]
        return results

    def get_directory_size(self, team_id: int, path: str) -> int:
        """
        Get directory size in the Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param path: Path to Directory.
        :type path: str
        :return: Directory size in the Team Files
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 9
            path = "/My_App_Test/"
            size = api.file.get_directory_size(team_id, path)

            print(size)
            # Output: 3478687
        """
        dir_size = 0
        file_infos = self.list2(team_id, path)
        for file_info in file_infos:
            dir_size += file_info.sizeb
        return dir_size

    def _download(self, team_id, remote_path, local_save_path, progress_cb=None):  # TODO: progress bar
        response = self._api.post('file-storage.download', {ApiField.TEAM_ID: team_id, ApiField.PATH: remote_path}, stream=True)
        #print(response.headers)
        #print(response.headers['Content-Length'])
        ensure_base_path(local_save_path)
        with open(local_save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def download(self, team_id: int, remote_path: str, local_save_path: str, cache: Optional[FileCache] = None,
                 progress_cb: Optional[Progress] = None) -> None:
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
        :type progress_cb: Progress, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_file = "/999_App_Test/ds1/01587.json"
            local_save_path = "/home/admin/Downloads/01587.json"

            api.file.download(8, path_to_file, local_save_path)
        """
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
                        raise KeyError(f"Remote and local hashes are different (team id: {team_id}, file: {remote_path})")
                    cache.write_object(local_save_path, file_info.hash)
                else:
                    cache.read_object(file_info.hash, local_save_path)
                    if progress_cb is not None:
                        progress_cb(get_file_size(local_save_path))

    def download_directory(self, team_id: int, remote_path: str, local_save_path: str, progress_cb: Optional[Progress]=None) -> None:
        """
        Download Directory from Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param remote_path: Path to Directory in Team Files.
        :type remote_path: str
        :param local_save_path: Local save path.
        :type local_save_path: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_save_path = "/home/admin/Downloads/My_local_test"

            api.file.download_directory(9, path_to_dir, local_save_path)
        """
        local_temp_archive = os.path.join(local_save_path, "temp.tar")
        self._download(team_id, remote_path, local_temp_archive, progress_cb)
        tr = tarfile.open(local_temp_archive)
        tr.extractall(local_save_path)
        silent_remove(local_temp_archive)
        temp_dir = os.path.join(local_save_path, os.path.basename(os.path.normpath(remote_path)))
        file_names = os.listdir(temp_dir)
        for file_name in file_names:
            shutil.move(os.path.join(temp_dir, file_name), local_save_path)
        shutil.rmtree(temp_dir)


    def _upload_legacy(self, team_id, src, dst):
        def path_to_bytes_stream(path):
            return open(path, 'rb')
        item = get_file_name_with_ext(dst)
        content_dict = {}
        content_dict[ApiField.NAME] = item

        dst_dir = os.path.dirname(dst)
        if not dst_dir.endswith(os.path.sep):
            dst_dir += os.path.sep
        content_dict[ApiField.PATH] = dst_dir # os.path.basedir ...
        content_dict["file"] = (item, path_to_bytes_stream(src), mimetypes.MimeTypes().guess_type(src)[0])
        encoder = MultipartEncoder(fields=content_dict)
        resp = self._api.post("file-storage.upload?teamId={}".format(team_id), encoder)
        return resp.json()

    def upload(self, team_id: int, src: str, dst: str, progress_cb: Optional[Progress] = None) -> NamedTuple:
        """
        Upload File to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file path.
        :type src: str
        :param dst: Path to File in Team Files.
        :type dst: str
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :return: Information about File. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_path = "/home/admin/Downloads/01587.json"
            dst_remote_path = "/999_App_Test/ds1/01587.json"

            api.file.upload(8, src_path, dst_remote_path)
        """
        return self.upload_bulk(team_id, [src], [dst], progress_cb)[0]

    def upload_bulk(self, team_id: int, src_paths: List[str], dst_paths: List[str], progress_cb: Optional[Progress] = None) -> List[NamedTuple]:
        """
        Upload Files to Team Files.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param src: Local source file paths.
        :type src: List[str]
        :param dst: Destination paths for Files to Team Files.
        :type dst: List[str]
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :return: Information about Files. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_paths = ["/home/admin/Downloads/01587.json", "/home/admin/Downloads/01588.json","/home/admin/Downloads/01589.json"]
            dst_remote_paths = ["/999_App_Test/ds1/01587.json", "/999_App_Test/ds1/01588.json", "/999_App_Test/ds1/01589.json"]

            api.file.upload_bulk(8, src_paths, dst_remote_paths)
        """
        def path_to_bytes_stream(path):
            return open(path, 'rb')
        content_dict = []
        for idx, (src, dst) in enumerate(zip(src_paths, dst_paths)):
            name = get_file_name_with_ext(dst)
            content_dict.append((ApiField.NAME, name))
            dst_dir = os.path.dirname(dst)
            if not dst_dir.endswith(os.path.sep):
                dst_dir += os.path.sep
            content_dict.append((ApiField.PATH, dst_dir))
            content_dict.append(("file", (name, path_to_bytes_stream(src), mimetypes.MimeTypes().guess_type(src)[0])))
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # NotImplementedError('Method is not supported')
        """
        pass

    def remove(self, team_id: int, path: str) -> None:
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.file.remove(8, "/999_App_Test/ds1/01587.json")
        """
        resp = self._api.post("file-storage.remove",{ApiField.TEAM_ID: team_id, ApiField.PATH: path})

    def exists(self, team_id: int, remote_path: str) -> bool:
        """
         Checks if file exists in Team Files.

         :param team_id: Team ID in Supervisely.
         :type team_id: int
         :param remote_path: Remote path to File in Team Files.
         :type remote_path: str
         :return: True if file exists, otherwise False
         :rtype: :class:`bool`
         :Usage example:

          .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file = api.file.exists(8, "/999_App_Test/ds1/02163.json") # True
            file = api.file.exists(8, "/999_App_Test/ds1/01587.json") # False
         """
        path_infos = self.list(team_id, remote_path)
        for info in path_infos:
            if info["path"] == remote_path:
                return True
        return False

    def dir_exists(self, team_id: int, remote_directory: str) -> bool:
        """
         Checks if directory exists in Team Files.

         :param team_id: Team ID in Supervisely.
         :type team_id: int
         :param remote_path: Remote path to directory in Team Files.
         :type remote_path: str
         :return: True if directory exists, otherwise False
         :rtype: :class:`bool`
         :Usage example:

          .. code-block:: python

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file = api.file.exists(8, "/999_App_Test/")   # True
            file = api.file.exists(8, "/10000_App_Test/") # False
         """
        files_infos = self.list(team_id, remote_directory)
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

        def _combine(suffix:int=None):
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            file_id = 7660
            file_url = sly.api.file.get_url(file_id)
            print(file_url)
            # Output: http://supervise.ly/files/7660
         """
        return urllib.parse.urljoin(self._api.server_address, "files/{}".format(file_id))

    def get_info_by_path(self, team_id: int, remote_path: str) -> NamedTuple:
        """
         Gets File information by path in Team Files.

         :param team_id: Team ID in Supervisely.
         :type team_id: int
         :param remote_path: Remote path to file in Team Files.
         :type remote_path: str
         :return: Information about File. See :class:`info_sequence<info_sequence>`
         :rtype: :class:`NamedTuple`
         :Usage example:

          .. code-block:: python

             os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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
             #                  full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
         """
        path_infos = self.list(team_id, remote_path)
        for info in path_infos:
            if info["path"] == remote_path:
                return self._convert_json_info(info)
        return None

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        #if res.storage_path is not None:
        #    res = res._replace(full_storage_url=urllib.parse.urljoin(self._api.server_address, res.storage_path))
        return res

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
         Gets information about File by ID.

         :param id: File ID in Supervisely.
         :type id: int
         :return: Information about File. See :class:`info_sequence<info_sequence>`
         :rtype: :class:`NamedTuple`
         :Usage example:

          .. code-block:: python

             os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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
             #                  full_storage_url='http://supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/8/y/P/rn/...json')
         """
        resp = self._api.post('file-storage.info', {ApiField.ID: id})
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_dir_name = api.file.get_free_dir_name(9, "/My_App_Test")
            print(new_dir_name)
            # Output: /My_App_Test_001
         """
        res_dir = dir_path.rstrip('/')
        suffix = 1
        while self.dir_exists(team_id, res_dir):
            res_dir = dir_path.rstrip('/') + f"_{suffix:03d}"
            suffix += 1
        return res_dir

    def upload_directory(self, team_id: int, local_dir: str, remote_dir: str,
                         change_name_if_conflict: Optional[bool]=True, progress_size_cb: Optional[Progress] = None) -> str:
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            path_to_dir = "/My_App_Test/ds1"
            local_path = "/home/admin/Downloads/My_local_test"

            api.file.upload_directory(9, local_path, path_to_dir)
        """
        if self.dir_exists(team_id, remote_dir):
            if change_name_if_conflict is True:
                res_remote_dir = self.get_free_dir_name(team_id, remote_dir)
            else:
                raise FileExistsError(f"Directory {remote_dir} already exists in your team (id={team_id})")
        else:
            res_remote_dir = remote_dir

        local_files = list_files_recursively(local_dir)
        remote_files = [file.replace(local_dir, res_remote_dir) for file in local_files]

        upload_results = self.upload_bulk(team_id, local_files, remote_files, progress_size_cb)
        return res_remote_dir

