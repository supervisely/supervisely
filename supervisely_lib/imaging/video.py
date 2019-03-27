# coding: utf-8

import os


ALLOWED_VIDEO_EXTENSIONS = ['.avi', '.mp4']


class VideoExtensionError(Exception):
    pass


def is_allowed_video_extension(path: str) -> bool:
    file_ext = str(os.path.splitext(path)[1]).lower()
    return file_ext in ALLOWED_VIDEO_EXTENSIONS


def validate_video_extension(path: str):
    if not is_allowed_video_extension(path):
        raise VideoExtensionError('Wrong video file format. The following list of formats is supported: {0}'
                                  .format(ALLOWED_VIDEO_EXTENSIONS))

