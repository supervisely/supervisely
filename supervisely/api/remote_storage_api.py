# coding: utf-8
from supervisely.api.module_api import ModuleApiBase, ApiField
from supervisely.collection.str_enum import StrEnum
from supervisely.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder
import mimetypes


class Provider(StrEnum):
    """Provider"""

    S3 = "s3"
    """S3"""
    GOOGLE = "google"
    """GOOGLE"""
    AZURE = "azure"
    """AZURE"""
    FS = "fs"
    """FS"""

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
    """RemoteStorageApi"""

    def _convert_json_info(self, info: dict):
        """_convert_json_info"""
        return info

    def list(self, path: str, recursive: bool = True, files: bool = True, folders: bool = True, limit: int = 10000, start_after: str = "") -> dict:
        """
        List files and directories for given remote path.

        :param path: Remote path with items that you want to list.
        :type path: str
        :param recursive: List remote path revursively.
        :type recursive: bool
        :param files: List files in the given path.
        :type files: bool
        :param folders: List folders in the given path.
        :type folders: bool
        :param limit: Limit of files to list. 10000 is the maximum limit.
        :type limit: int
        :param start_after: Start listing path after given file name.
        :type start_after: str
        :returns: List of files in the given remote path
        :rtype: dict

        """
        
        Provider.validate_path(path)
        path = path.rstrip("/") + "/"
        resp = self._api.get(
            "remote-storage.list",
            {
                ApiField.PATH: path,
                "recursive": recursive,
                "files": files,
                "folders": folders,
                "limit": limit,
                "startAfter": start_after
            },
        )
        return resp.json()

    def download_path(self, remote_path: str, save_path: str, progress_cb=None):
        """
        Downloads item from given remote path to given local path.

        :param remote_path: Remote path to item that you want to upload.
        :type remote_path: str
        :param save_path: Local save path.
        :type save_path: str
        :Usage example:

        .. code-block:: python

            provider = "s3" # can be one of ["s3", "google", "azure"]
            bucket = "bucket-test-export"
            path_in_bucket = "/demo/image.jpg"
            remote_path = api.remote_storage.get_remote_path(provider, bucket, path_in_bucket)
            # or alternatively use this:
            # remote_path = f"{provider}://{bucket}{path_in_bucket}"
            api.remote_storage.upload_path(local_path="images/my-cats.jpg", remote_path=remote_path)
        """
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

    def upload_path(self, local_path: str, remote_path: str):
        """
        Uploads item from given local path to given remote path.

        :param local_path: Local path to item that you want to upload.
        :type local_path: str
        :param remote_path: Remote destination path.
        :type remote_path: str
        :Usage example:

        .. code-block:: python

            provider = "s3" # can be one of ["s3", "google", "azure"]
            bucket = "bucket-test-export"
            path_in_bucket = "/demo/image.jpg"
            remote_path = api.remote_storage.get_remote_path(provider, bucket, path_in_bucket)
            # or alternatively use this:
            # remote_path = f"{provider}://{bucket}{path_in_bucket}"
            api.remote_storage.upload_path(local_path="images/my-cats.jpg", remote_path=remote_path)
        """
        Provider.validate_path(remote_path)
        return self._upload_paths_batch([local_path], [remote_path])

    def _upload_paths_batch(self, local_paths, remote_paths):
        """_upload_paths_batch"""
        
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

    def get_remote_path(self, provider: str, bucket: str, path_in_bucket: str) -> str:
        """
        Returns remote path.

        :param provider: Can be one of "s3", "google", "azure".
        :type provider: str
        :param bucket: Name of the bucket container.
        :type bucket: str
        :param path_in_bucket: Path to item in bucket.
        :type path_in_bucket: str
        :Usage example:

        .. code-block:: python

            provider = "s3"
            bucket = "bucket-test-export"
            path_in_bucket = "/demo/image.jpg"
            remote_path = api.remote_storage.get_remote_path(provider, bucket, path_in_bucket)
            # Output: s3://bucket-test-export/demo/image.jpg
        """
        res_path = f"{provider}://{bucket}/{path_in_bucket.lstrip('/')}"
        Provider.validate_path(res_path)
        return res_path
