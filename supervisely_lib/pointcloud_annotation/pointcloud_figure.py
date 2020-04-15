# coding: utf-8

from supervisely_lib._utils import take_with_default
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.video_annotation.video_figure import VideoFigure
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection


class PointcloudFigure(VideoFigure):
    def __init__(self, parent_object, geometry, frame_index=None, key=None):
        super().__init__(parent_object, geometry, frame_index, key)
        #@TODO: validate geometry - allowed: only cuboid_3d

    def validate_bounds(self, img_size, _auto_correct=False):
        raise NotImplementedError()
