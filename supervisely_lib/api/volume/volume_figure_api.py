# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.api.video.video_figure_api import VideoFigureApi
import supervisely_lib.volume_annotation.constants as const

class VolumeFigureApi(VideoFigureApi):
    def create(self, volume_id, object_id, slice_index, normal, geometry_json, geometry_type):
        return super().create(volume_id,
                              object_id,
                              {ApiField.META: {const.INDEX: slice_index, const.NORMAL: normal}},
                              geometry_json,
                              geometry_type)
