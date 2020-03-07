# coding: utf-8

import os

# Do NOT use directly for video extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VIDEO_EXTENSIONS = ['.avi', '.mkv', '.mp4']


class VideoExtensionError(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    if not is_valid_ext(ext):
        raise VideoExtensionError('Unsupported video extension: {}. Only the following extensions are supported: {}.'
                                  .format(ALLOWED_VIDEO_EXTENSIONS))

