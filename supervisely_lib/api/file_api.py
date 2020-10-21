# coding: utf-8
import os
from pathlib import Path
import urllib
from supervisely_lib.api.module_api import ModuleApiBase, ApiField
from supervisely_lib.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder
import mimetypes
from supervisely_lib.io.fs import get_file_ext, get_file_name


class FileApi(ModuleApiBase):
    def list(self, team_id, path):
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        return response.json()

    def download(self, team_id, remote_path, local_save_path):
        response = self._api.post('file-storage.download', {ApiField.TEAM_ID: team_id, ApiField.PATH: remote_path}, stream=True)
        ensure_base_path(local_save_path)
        with open(local_save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def upload(self, team_id, src, dst):
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
    # Example
    # {
    #     "id": 38292,
    #     "storagePath": "teams_storage/1/X/S/rB/DrtCQEniZRAnj7oxAyJCrF80ViCC6swBcG6hYlUwkjlc0dE58lmIhRvGW00JSrQKO1s5NRuqaIAUZUUU50vK3vp09E62vCCErUF6owvkauzncYMtHssgXqoi9rGY.txt",
    #     "path": "/reports/classes_stats/max/2020-09-23-12:26:33_pascal voc 2012_(id_355).lnk",
    #     "userId": 1,
    #     "meta": {
    #         "size": 44,
    #         "mime": "text/plain",
    #         "ext": "lnk"
    #     },
    #     "name": "2020-09-23-12:26:33_pascal voc 2012_(id_355).lnk",
    #     "teamId": 1
    # }

    def rename(self, old_name, new_name):
        pass

    def remove(self, team_id, path):
        resp = self._api.post("file-storage.remove",{ApiField.TEAM_ID: team_id, ApiField.PATH: path})

    def exists(self, team_id, remote_path):
        path_infos = self.list(team_id, remote_path)
        for info in path_infos:
            if info["path"] == remote_path:
                return True
        return False

    def get_free_name(self, team_id, path):
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

    def get_url(self, file_id):
        return urllib.parse.urljoin(self._api.server_address, "files/{}".format(file_id))