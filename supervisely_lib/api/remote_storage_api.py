# coding: utf-8
from supervisely_lib.api.module_api import ModuleApiBase, ApiField
from supervisely_lib.collection.str_enum import StrEnum
from supervisely_lib.io.fs import ensure_base_path


class Provider(StrEnum):
    S3 = 's3'
    GOOGLE = 'google'
    AZURE = 'azure'

    @staticmethod
    def validate_path(path):
        if not path.startswith(str(Provider.S3)) and \
            not path.startswith(str(Provider.GOOGLE)) and \
            not path.startswith(str(Provider.AZURE)): \
            raise ValueError("Incorrect cloud path, learn more here: https://docs.supervise.ly/enterprise-edition/advanced-tuning/s3#links-plugin-cloud-providers-support")


class RemoteStorageApi(ModuleApiBase):
    def _convert_json_info(self, info: dict):
        return info

    def list(self, path, recursive=True, files=True, folders=True):
        Provider.validate_path(path)
        resp = self._api.get('remote-storage.list', {
            ApiField.PATH: path,
            'recursive': recursive,
            'files': files,
            'folders': folders
        })
        return resp.json()

    def download_path(self, remote_path, save_path, progress_cb=None):
        ensure_base_path(save_path)
        response = self._api.post('remote-storage.download', {ApiField.LINK: remote_path}, stream=True)
        # if "Content-Length" in response.headers:
        #     length = int(response.headers['Content-Length'])
        with open(save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))
