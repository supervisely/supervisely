# coding: utf-8

from supervisely_lib.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class ObjectApi(RemoveableBulkModuleApi):
    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.DESCRIPTION,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT,
                ApiField.DATASET_ID,
                ApiField.CLASS_ID,
                ApiField.ENTITY_ID,
                ApiField.TAGS,
                ApiField.META,
                ApiField.CREATED_BY_ID,
                ]

    @staticmethod
    def info_tuple_name():
        return 'ObjectInfo'

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'objects.info')

    def get_list(self, dataset_id, filters=None):
        return self.get_list_all_pages('annotation-objects.list',  {ApiField.DATASET_ID: dataset_id,
                                                                    ApiField.FILTER: filters or []})

    def _append_bulk(self, tag_api, entity_id, project_id, dataset_id, objects, key_id_map: KeyIdMap = None, is_pointcloud=False):
        if len(objects) == 0:
            return []

        objcls_name_id_map = self._api.object_class.get_name_to_id_map(project_id)

        items = []
        for obj in objects:
            new_obj = {ApiField.CLASS_ID: objcls_name_id_map[obj.obj_class.name]}

            if not is_pointcloud:
            #if entity_id is not None:
                new_obj[ApiField.ENTITY_ID] = entity_id
            items.append(new_obj)

        response = self._api.post('annotation-objects.bulk.add', {ApiField.DATASET_ID: dataset_id,
                                                                  ApiField.ANNOTATION_OBJECTS: items})
        ids = [obj[ApiField.ID] for obj in response.json()]
        KeyIdMap.add_objects_to(key_id_map, [obj.key() for obj in objects], ids)

        # add tags to objects
        tag_api.append_to_objects(entity_id, project_id, objects, key_id_map)

        return ids
