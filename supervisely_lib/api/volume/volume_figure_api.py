# coding: utf-8
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
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

    def append_bulk(self, volume_id, figures, plane_name, key_id_map: KeyIdMap):
        keys = []
        figures_json = []
        normal = const.PLANE_NORMALS[plane_name]
        for figure in figures:
            keys.append(figure.key())
            fig_json = figure.to_json(key_id_map, save_meta=True)

            slice_index = fig_json[ApiField.META][ApiField.FRAME]
            fig_json[ApiField.META] = {const.INDEX: slice_index, const.NORMAL: normal}
            figures_json.append(fig_json)

        self._append_bulk(volume_id, figures_json, keys, key_id_map)
