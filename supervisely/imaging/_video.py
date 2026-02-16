# coding: utf-8

import os

# Do NOT use directly for video extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VIDEO_EXTENSIONS = ['.avi', '.mkv', '.mp4']


class VideoExtensionError(Exception):
    """Video extension error."""
    pass


def is_valid_ext(ext: str) -> bool:
    """
    The function is_valid_ext checks file extension for list of supported video extensions('.avi', '.mp4')
    :param ext: file extention
    :returns: True if file extention in list of supported images extensions, False - in otherwise
    :rtype: bool

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            sly.imaging._video.is_valid_ext(".mp4")  # True
            sly.imaging._video.is_valid_ext(".jpeg") # False
    """
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    The function has_valid_ext checks if a given file has a supported extension('.avi', '.mp4')
    :param path: the path to the input file
    :returns: True if a given file has a supported extension, False - in otherwise
    :rtype: bool

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            sly.imaging._video.has_valid_ext("/home/root/video/video.mp4")  # True
            sly.imaging._video.has_valid_ext("/home/root/video/video.jpeg")  # False
    """
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str):
    """
    The function validate_ext generate exception error if file extention is not in list of supported videos
    extensions('.avi', '.mp4')

    :param ext: file extention
    :type ext: str
    :returns: None
    :rtype: None

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            sly.imaging._video.validate_ext(".jpeg")
    """
    if not is_valid_ext(ext):
        raise VideoExtensionError('Unsupported video extension: {}. Only the following extensions are supported: {}.'
                                  .format(ext, ALLOWED_VIDEO_EXTENSIONS))
