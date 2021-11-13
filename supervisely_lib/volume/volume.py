# coding: utf-8

import os


# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VOLUME_EXTENSIONS = ['.nrrd']


class VolumeExtensionError(Exception):
    pass


class UnsupportedVolumeFormat(Exception):
    pass


class VolumeReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    '''
    Checks if given extention is supported
    :param ext: str
    :return: bool
    '''
    return ext.lower() in ALLOWED_VOLUME_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    '''
    Checks if file from given path with given extention is supported
    :param path: str
    :return: bool
    '''
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    '''
    Raise error if given extention is not supported
    :param ext: str
    '''
    if not is_valid_ext(ext):
        raise UnsupportedVolumeFormat('Unsupported Volume extension: {}. Only the following extensions are supported: {}.'
                                      .format(ext, ALLOWED_VOLUME_EXTENSIONS))


def validate_format(path):
    #@TODO: later
    validate_ext(path)
