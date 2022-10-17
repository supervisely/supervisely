# coding: utf-8

import os
from supervisely._utils import is_development, abs_url

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_POINTCLOUD_EXTENSIONS = ['.pcd']


class PointcloudExtensionError(Exception):
    pass


class UnsupportedPointcloudFormat(Exception):
    pass


class PointcloudReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extention is supported
    :param ext: str
    :return: bool
    """
    return ext.lower() in ALLOWED_POINTCLOUD_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if file from given path with given extention is supported
    :param path: str
    :return: bool
    """
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str) -> None:
    """
    Raise error if given extention is not supported
    :param ext: str
    """
    if not is_valid_ext(ext):
        raise UnsupportedPointcloudFormat('Unsupported pointcloud extension: {}. Only the following extensions are supported: {}.'
                                          .format(ext, ALLOWED_POINTCLOUD_EXTENSIONS))


def validate_format(path):
    _, ext = os.path.splitext(path)
    validate_ext(ext)


def get_labeling_tool_url(dataset_id, pointcloud_id):
    res = f"/app/point-clouds/?datasetId={dataset_id}&pointCloudId={pointcloud_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'
