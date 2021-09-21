# coding: utf-8
import os
import shutil
import tarfile
from pathlib import Path
import urllib
from supervisely_lib.api.module_api import ModuleApiBase, ApiField
from supervisely_lib.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import mimetypes
from supervisely_lib.io.fs import get_file_ext, get_file_name, list_files_recursively
from supervisely_lib.imaging.image import write_bytes, get_hash
from supervisely_lib.task.progress import Progress
from supervisely_lib.io.fs_cache import FileCache
from supervisely_lib.io.fs import get_file_hash, get_file_ext, get_file_size, list_files_recursively, silent_remove

class FileApi(ModuleApiBase):
    @staticmethod
    def info_sequence():
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
        return 'FileInfo'

    def list(self, team_id, path):
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        return response.json()

    def list2(self, team_id, path):
        response = self._api.post('file-storage.list', {ApiField.TEAM_ID: team_id, ApiField.PATH: path})
        results = [self._convert_json_info(info_json) for info_json in response.json()]
        return results

    def get_directory_size(self, team_id, path):
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

    def download(self, team_id, remote_path, local_save_path, cache: FileCache = None, progress_cb=None):
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

    def download_directory(self, team_id, remote_path, local_save_path, progress_cb=None):
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

    def upload(self, team_id, src, dst, progress_cb=None):
        return self.upload_bulk(team_id, [src], [dst], progress_cb)[0]

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

    def dir_exists(self, team_id, remote_directory):
        files_infos = self.list(team_id, remote_directory)
        if len(files_infos) > 0:
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

    def get_free_dir_name(self, team_id, dir_path):
        res_dir = dir_path.rstrip('/')
        suffix = 1
        while self.dir_exists(team_id, res_dir):
            res_dir = dir_path.rstrip('/') + f"_{suffix:03d}"
            suffix += 1
        return res_dir

    def upload_directory(self, team_id, local_dir, remote_dir,
                         change_name_if_conflict=True, progress_size_cb=None):
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

