# coding: utf-8

import os


# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_POINTCLOUD_EXTENSIONS = ['.pcd']


class PointcloudExtensionError(Exception):
    pass


class UnsupportedPointcloudFormat(Exception):
    pass


class PointcloudReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    return ext.lower() in ALLOWED_POINTCLOUD_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    if not is_valid_ext(ext):
        raise UnsupportedPointcloudFormat('Unsupported pointcloud extension: {}. Only the following extensions are supported: {}.'
                                          .format(ext, ALLOWED_POINTCLOUD_EXTENSIONS))


def validate_format(path):
    #@TODO: later
    validate_ext(path)
