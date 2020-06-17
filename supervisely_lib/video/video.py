# coding: utf-8

import os
import skvideo.io
from supervisely_lib import logger as default_logger
from supervisely_lib.io.fs import get_file_name, get_file_ext
from supervisely_lib import rand_str


# Do NOT use directly for video extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VIDEO_EXTENSIONS = ['.avi', '.mp4', '.3gp', '.flv', '.webm', '.wmv', '.mov', '.mkv']


_SUPPORTED_CONTAINERS = {'mp4', 'webm', 'ogg', 'ogv'}
_SUPPORTED_CODECS = {'h264', 'vp8', 'vp9'}


class VideoExtensionError(Exception):
    pass


class UnsupportedVideoFormat(Exception):
    pass


class VideoReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    if not is_valid_ext(ext):
        raise UnsupportedVideoFormat('Unsupported video extension: {}. Only the following extensions are supported: {}.'
                                     .format(ext, ALLOWED_VIDEO_EXTENSIONS))


def get_image_size_and_frames_count(path):
    vreader = skvideo.io.FFmpegReader(path)
    vlength = vreader.getShape()[0]
    img_height = vreader.getShape()[1]
    img_width = vreader.getShape()[2]

    img_size = (img_height, img_width)

    return img_size, vlength


def validate_format(path):
    try:
        get_image_size_and_frames_count(path)
    except Exception as e:
        raise VideoReadException(
            'Error has occured trying to read video {!r}. Original exception message: {!r}'.format(path, str(e)))

    validate_ext(os.path.splitext(path)[1])


def _check_video_requires_processing(video_info, stream_info):
    need_process_container = True
    for name in video_info["meta"]["formatName"].split(','):
        name = name.strip().split('.')[-1]
        if name in _SUPPORTED_CONTAINERS:
            need_process_container = False
            break

    need_process_codec = True
    codec = stream_info["codecName"]
    if codec in _SUPPORTED_CODECS:
        need_process_codec = False

    if (need_process_container is False) and (need_process_codec is False):
        return False

    return True


def count_video_streams(all_streams):
    count = 0
    for stream_info in all_streams:
        if stream_info["codecType"] == "video":
            count += 1
    return count


def get_video_streams(all_streams):
    video_streams = []
    for stream_info in all_streams:
        if stream_info["codecType"] == "video":
            video_streams.append(stream_info)
    return video_streams


def warn_video_requires_processing(file_name, logger=None):
    if logger is None:
        logger = default_logger
    logger.warning("Video Stream {!r} is skipped: requires transcoding. Transcoding is supported only in Enterprise Edition (EE)".format(file_name))


def gen_video_stream_name(file_name, stream_index):
    return "{}_stream_{}_{}{}".format(get_file_name(file_name), stream_index, rand_str(5), get_file_ext(file_name))
