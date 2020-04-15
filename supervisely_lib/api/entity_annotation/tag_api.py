# coding: utf-8

from supervisely_lib.api.module_api import ModuleApi
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class TagApi(ModuleApi):
    _entity_id_field = None
    _method_bulk_add = None

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                ApiField.PROJECT_ID,
                ApiField.NAME,
                ApiField.SETTINGS,
                ApiField.COLOR,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        return 'TagInfo'

    def get_list(self, project_id, filters=None):
        return self.get_list_all_pages('tags.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})

    def get_name_to_id_map(self, project_id):
        tags_info = self.get_list(project_id)
        return {tag_info.name: tag_info.id for tag_info in tags_info}

    def _tags_to_json(self, tags: KeyIndexedCollection, tag_name_id_map=None, project_id=None):
        if tag_name_id_map is None and project_id is None:
            raise RuntimeError("Impossible to get ids for project tags")
        if tag_name_id_map is None:
            tag_name_id_map = self.get_name_to_id_map(project_id)
        tags_json = []
        tags_keys = []
        for tag in tags:
            tag_json = tag.to_json()
            tag_json[ApiField.TAG_ID] = tag_name_id_map[tag.name]
            tags_json.append(tag_json)
            tags_keys.append(tag.key())
        return tags_json, tags_keys

    def append_to_entity(self, entity_id, project_id, tags: KeyIndexedCollection, key_id_map: KeyIdMap = None):
        if len(tags) == 0:
            return []
        tags_json, tags_keys = self._tags_to_json(tags, project_id=project_id)
        ids = self._append_json(entity_id, tags_json)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)
        return ids

    def _append_json(self, entity_id, tags_json):
        if self._method_bulk_add is None:
            raise RuntimeError("self._method_bulk_add is not defined in child class")
        if self._entity_id_field is None:
            raise RuntimeError("self._entity_id_field is not defined in child class")

        if len(tags_json) == 0:
            return []
        response = self._api.post(self._method_bulk_add, {self._entity_id_field: entity_id, ApiField.TAGS: tags_json})
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids

    def append_to_objects(self, entity_id, project_id, objects: KeyIndexedCollection, key_id_map: KeyIdMap):
        tag_name_id_map = self.get_name_to_id_map(project_id)

        tags_to_add = []
        tags_keys = []
        for object in objects:
            obj_id = key_id_map.get_object_id(object.key())
            if obj_id is None:
                raise RuntimeError("Can not add tags to object: OBJECT_ID not found for key {}".format(object.key()))
            tags_json, cur_tags_keys = self._tags_to_json(object.tags, tag_name_id_map=tag_name_id_map)
            for tag in tags_json:
                tag[ApiField.OBJECT_ID] = obj_id
                tags_to_add.append(tag)
            tags_keys.extend(cur_tags_keys)

        if len(tags_keys) != len(tags_to_add):
            raise RuntimeError("SDK error: len(tags_keys) != len(tags_to_add)")
        if len(tags_keys) == 0:
            return
        ids = self.append_to_objects_json(entity_id, tags_to_add)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)

    def append_to_objects_json(self, entity_id, tags_json):
        if len(tags_json) == 0:
            return []
        response = self._api.post('annotation-objects.tags.bulk.add', {ApiField.ENTITY_ID: entity_id, ApiField.TAGS: tags_json})
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids


