import imghdr
import os

from supervisely import logger
from supervisely.imaging.image import is_valid_ext as is_valid_image_ext
from supervisely.io.fs import get_file_ext
from supervisely.pointcloud.pointcloud import is_valid_ext as is_valid_point_cloud_ext
from supervisely.video.video import is_valid_ext as is_valid_video_ext
from supervisely.volume.volume import is_valid_ext as is_valid_volume_ext

# @TODO:
# [ ] exclude annotations and junk files
# [ ] check img extensions in imghdr


possible_annotations_exts = [".json", ".xml", ".txt"]
possible_junk_exts = [".DS_Store"]


def contains_only_images(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            ext = get_file_ext(full_path)
            if ext in possible_annotations_exts or ext in possible_junk_exts:  # add better check
                continue
            if imghdr.what(full_path) is None:
                logger.info(f"Non-image file found: {full_path}")
                return False
    return True


def contains_only_videos(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = get_file_ext(file)
            if is_valid_video_ext(ext):
                logger.info(f"Non-video file found: {os.path.join(root, file)}")
                return False
    return True


def contains_only_point_clouds(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = get_file_ext(file)
            if is_valid_point_cloud_ext(ext):
                logger.info(f"Non-point cloud file found: {os.path.join(root, file)}")
                return False
    return True


def contains_only_volumes(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = get_file_ext(file)
            if is_valid_volume_ext(ext):
                logger.info(f"Non-volume file found: {os.path.join(root, file)}")
                return False
    return True
