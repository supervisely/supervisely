# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class FigureApi(RemoveableBulkModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.UPDATED_AT,
                ApiField.CREATED_AT,
                ApiField.ENTITY_ID,
                ApiField.OBJECT_ID,
                ApiField.PROJECT_ID,
                ApiField.DATASET_ID,
                ApiField.FRAME_INDEX,
                ApiField.GEOMETRY_TYPE,
                ApiField.GEOMETRY
                ]

    @staticmethod
    def info_tuple_name():
        return 'FigureInfo'

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: figure metadata by numeric id
        '''
        return self._get_info_by_id(id, 'figures.info')

    def create(self, entity_id, object_id, meta, geometry_json, geometry_type, track_id=None):
        input_figure = {
                    ApiField.META: meta,
                    ApiField.OBJECT_ID: object_id,
                    ApiField.GEOMETRY_TYPE: geometry_type,
                    ApiField.GEOMETRY: geometry_json};

        if track_id is not None:
            input_figure[ApiField.TRACK_ID] = track_id

        body = {ApiField.ENTITY_ID: entity_id,
                ApiField.FIGURES: [input_figure]}

        response = self._api.post('figures.bulk.add', body)
        return response.json()[0][ApiField.ID]

    def get_by_ids(self, dataset_id, ids):
        '''
        :param dataset_id: int
        :param ids: list of integers
        :return: list of figures with given ids from dataset with given id
        '''
        filters = [{"field": "id", "operator": "in", "value": ids}]
        figures_infos = self.get_list_all_pages('figures.list', {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters})

        if len(ids) != len(figures_infos):
            ids_downloaded = [info.id for info in figures_infos]
            raise RuntimeError("Ids don't exist on server: {}".format(set(ids_downloaded) - set(ids)))

        id_to_item = {info.id: info for info in figures_infos}

        figures = []
        for input_id in ids:
            figures.append(id_to_item[input_id])

        return figures

    def _append_bulk(self, entity_id, figures_json, figures_keys, key_id_map: KeyIdMap):
        if len(figures_json) == 0:
            return
        resp = self._api.post('figures.bulk.add', {ApiField.ENTITY_ID: entity_id, ApiField.FIGURES: figures_json})
        for key, resp_obj in zip(figures_keys, resp.json()):
            figure_id = resp_obj[ApiField.ID]
            key_id_map.add_figure(key, figure_id)
