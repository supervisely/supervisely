# coding: utf-8
from supervisely.api.module_api import ModuleApiBase, ApiField
from supervisely.collection.str_enum import StrEnum
from supervisely.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder
import mimetypes


class Provider(StrEnum):
    S3 = "s3"
    GOOGLE = "google"
    AZURE = "azure"
    FS = "fs"

    @staticmethod
    def validate_path(path):
        if (
            not path.startswith(str(Provider.S3))
            and not path.startswith(str(Provider.GOOGLE))
            and not path.startswith(str(Provider.AZURE))
            and not path.startswith(str(Provider.FS))
        ):
            raise ValueError(
                "Incorrect cloud path, learn more here: https://docs.supervise.ly/enterprise-edition/advanced-tuning/s3#links-plugin-cloud-providers-support"
            )


class RemoteStorageApi(ModuleApiBase):
    def _convert_json_info(self, info: dict):
        return info

    def list(self, path, recursive=True, files=True, folders=True):
        Provider.validate_path(path)
        resp = self._api.get(
            "remote-storage.list",
            {
                ApiField.PATH: path,
                "recursive": recursive,
                "files": files,
                "folders": folders,
            },
        )
        return resp.json()

    def download_path(self, remote_path, save_path, progress_cb=None):
        Provider.validate_path(remote_path)
        ensure_base_path(save_path)
        response = self._api.post(
            "remote-storage.download", {ApiField.LINK: remote_path}, stream=True
        )
        # if "Content-Length" in response.headers:
        #     length = int(response.headers['Content-Length'])
        with open(save_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))

    def upload_path(self, local_path, remote_path):
        """
        Usage example:
            provider = "s3" # can be one of ["s3", "google", "azure"]
            bucket_name = "bucket-test-export"
            path_in_bucket = "/demo/image.jpg"
            remote_path = f"{provider}://{bucket_name}{path_in_bucket}"
            api.remote_storage.upload_path(local_path="images/my-cats.jpg", remote_path=remote_path)
        """
        Provider.validate_path(remote_path)
        return self._upload_paths_batch([local_path], [remote_path])

    def _upload_paths_batch(self, local_paths, remote_paths):
        if len(local_paths) != len(remote_paths):
            raise ValueError(
                "Inconsistency in paths, len(local_paths) != len(remote_paths)"
            )
        if len(local_paths) == 0:
            return {}

        def path_to_bytes_stream(path):
            return open(path, "rb")

        content = []
        for idx, (src, dst) in enumerate(zip(local_paths, remote_paths)):
            content.append((ApiField.PATH, dst))
            name = get_file_name_with_ext(dst)
            content.append(
                (
                    "file",
                    (
                        name,
                        path_to_bytes_stream(src),
                        mimetypes.MimeTypes().guess_type(src)[0],
                    ),
                )
            )
        encoder = MultipartEncoder(fields=content)
        resp = self._api.post("remote-storage.upload", encoder)
        return resp.json()
