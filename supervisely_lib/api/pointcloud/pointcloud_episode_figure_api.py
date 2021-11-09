from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.api.entity_annotation.figure_api import FigureApi
from supervisely_lib.api.module_api import ApiField
from supervisely_lib._utils import batched


class PointcloudEpisodeFigureAPI(FigureApi):
    def create(self, entity_id, object_id, meta, geometry_json, geometry_type, track_id=None):
        raise RuntimeError('Not Available for PointcloudEpisodeFigureApi')

    def append_bulk(self, dataset_id, figures, pointcloud_ids, key_id_map: KeyIdMap):
        keys = []
        figures_json = []
        for figure, pointcloud_id in zip(figures, pointcloud_ids):
            keys.append(figure.key())
            figure_json = figure.to_json(key_id_map)
            figure_json[ApiField.ENTITY_ID] = pointcloud_id
            figures_json.append(figure_json)

        self._append_bulk(dataset_id, figures_json, keys, key_id_map)

    def _append_bulk(self, dataset_id, figures_json, figures_keys, key_id_map: KeyIdMap):
        if len(figures_json) == 0:
            return

        for (batch_keys, batch_jsons) in zip(batched(figures_keys, batch_size=100),
                                             batched(figures_json, batch_size=100)):
            resp = self._api.post('figures.bulk.add', {ApiField.DATASET_ID: dataset_id, ApiField.FIGURES: batch_jsons})
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)

    def _convert_json_info(self, info: dict, skip_missing=True):
        return super()._convert_json_info(info, skip_missing)