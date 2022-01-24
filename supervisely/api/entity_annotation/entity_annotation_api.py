# coding: utf-8

from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.video_annotation.key_id_map import KeyIdMap


class EntityAnnotationAPI(ModuleApi):
    _method_download_bulk = None
    _entity_ids_str = None

    def download(self, entity_id):
        raise NotImplementedError()

    def _download(self, dataset_id, entity_id):
        '''
        :param dataset_id: int
        :param entity_id: int
        :return: list of content(annotation with given id from dataset with given id), received after execution post request
        '''
        return self.download_bulk(dataset_id, [entity_id])[0]

    def download_bulk(self, dataset_id, entity_ids):
        '''
        :param dataset_id: int
        :param entity_ids: list of integers
        :return: list of content(annotations with given ids from dataset with given id), received after execution post request
        '''
        response = self._api.post(self._method_download_bulk, {ApiField.DATASET_ID: dataset_id,
                                                               self._entity_ids_str: entity_ids})
        return response.json()

    def _append(self, tag_api, object_api, figure_api, project_id, dataset_id, entity_id, tags, objects, figures, key_id_map: KeyIdMap = None):
        if key_id_map is None:
            # create for internal purposes (to link figures and tags to objects)
            key_id_map = KeyIdMap()

        tag_api.append_to_entity(entity_id, project_id, tags, key_id_map=key_id_map)
        object_api.append_bulk(entity_id, objects, key_id_map)
        figure_api.append_bulk(entity_id, figures, key_id_map)

    def append(self, entity_id, ann, key_id_map: KeyIdMap = None):
        raise NotImplementedError()

    # def append(self, video_id, ann, key_id_map: KeyIdMap = None):
    #     if key_id_map is None:
    #         # create for internal purposes (to link figures and tags to objects)
    #         key_id_map = KeyIdMap()
    #
    #     info = self._api.video.get_info_by_id(video_id)
    #
    #     self._api.tag.append_to_video(video_id, ann.tags, key_id_map=key_id_map)
    #     self._api.object.append_bulk(video_id, info.project_id, info.dataset_id, ann.objects, key_id_map)
    #     self._api.video.figure.append_bulk(video_id, ann.frames, key_id_map)
