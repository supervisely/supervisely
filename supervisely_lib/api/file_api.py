# coding: utf-8
import os
from pathlib import Path
import urllib
from supervisely_lib.api.module_api import ModuleApiBase, ApiField
from supervisely_lib.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import mimetypes
from supervisely_lib.io.fs import get_file_ext, get_file_name
from supervisely_lib.imaging.image import write_bytes, get_hash
from supervisely_lib.task.progress import Progress


class FileApi(ModuleApiBase):
    @staticmethod
    def info_sequence():
        return [
            ApiField.TEAM_ID,
            ApiField.ID,
            ApiField.USER_ID,
            ApiField.NAME,
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
        return 'FileInfo'

    def list(self, team_id, path):
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        return response.json()

    def download(self, team_id, remote_path, local_save_path):
        response = self._api.post('file-storage.download', {ApiField.TEAM_ID: team_id, ApiField.PATH: remote_path}, stream=True)
        ensure_base_path(local_save_path)
        with open(local_save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

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

    def upload(self, team_id, src, dst, progress_cb=None):
        return self.upload_bulk(team_id, [src], [dst], progress_cb)[0]

    # Example - return value
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

    def upload_bulk(self, team_id, src_paths, dst_paths, progress_cb=None):
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

    def get_info_by_path(self, team_id, remote_path):
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

    def get_info_by_id(self, id):
        resp = self._api.post('file-storage.info', {ApiField.ID: id})
        return self._convert_json_info(resp.json())