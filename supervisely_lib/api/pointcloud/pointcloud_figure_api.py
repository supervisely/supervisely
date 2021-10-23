# coding: utf-8

from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.api.entity_annotation.figure_api import FigureApi


class PointcloudFigureApi(FigureApi):
    def create(self, pointcloud_id, object_id, geometry_json, geometry_type, track_id=None):
        return super().create(pointcloud_id, object_id, {}, geometry_json, geometry_type, track_id)

    def append_bulk(self, pointcloud_id, figures, key_id_map: KeyIdMap):
        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map))

        self._append_bulk(pointcloud_id, figures_json, keys, key_id_map)

    def _convert_json_info(self, info: dict, skip_missing=True):
                return super()._convert_json_info(info, skip_missing)

