# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.api.entity_annotation.figure_api import FigureApi


class VideoFigureApi(FigureApi):
    def create(self, video_id, object_id, frame_index, geometry_json, geometry_type, track_id=None):
        return super().create(video_id,
                              object_id,
                              {ApiField.FRAME: frame_index},
                              geometry_json,
                              geometry_type,
                              track_id)

    def append_bulk(self, video_id, figures, key_id_map: KeyIdMap):
        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map, save_meta=True))

        self._append_bulk(video_id, figures_json, keys, key_id_map)

