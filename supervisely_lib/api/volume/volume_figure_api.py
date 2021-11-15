# coding: utf-8

from supervisely_lib.api.module_api import ApiField
from supervisely_lib.volume_annotation import constants as vol_const
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.api.entity_annotation.figure_api import FigureApi


class VolumeFigureApi(FigureApi):
    def append_bulk(self, volume_id, figures, key_id_map: KeyIdMap):
        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            fig_json = figure.to_json(key_id_map)
            figure_api_json = {
                ApiField.GEOMETRY: figure.geometry.to_json(),
                ApiField.GEOMETRY_TYPE: figure.geometry.geometry_name(),
                ApiField.META: figure.meta,
                ApiField.OBJECT_ID: fig_json[ApiField.OBJECT_ID],
                vol_const.PRIORITY: fig_json[vol_const.PRIORITY]
            }
            figures_json.append(figure_api_json)
        self._append_bulk(volume_id, figures_json, keys, key_id_map)
