# coding: utf-8
from __future__ import annotations
from typing import List, NamedTuple, Dict, Optional
from supervisely_lib.api.module_api import ModuleApi
from supervisely_lib.api.module_api import ApiField
from supervisely_lib.collection.key_indexed_collection import KeyIndexedCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


class TagApi(ModuleApi):
    """
    Tag object for :class:`VideoAnnotation<supervisely_lib.video_annotation.video_annotation.VideoAnnotation>`.
    """
    _entity_id_field = None
    _method_bulk_add = None

    @staticmethod
    def info_sequence():
        """
        NamedTuple ObjectInfo information about Object.

        :Example:

         .. code-block:: python

            TagInfo(id=29098692,
                    project_id=124976,
                    name='number_of_objects',
                    settings={'type': 'any_number',
                    'options': {'autoIncrement': False}},
                    color='#1F380F',
                    created_at='2021-03-23T13:25:34.705Z',
                    updated_at='2021-03-23T13:25:34.705Z')
        """
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

    def get_list(self, project_id: int, filters: Optional[List[Dict[str, str]]]=None) -> List[NamedTuple]:
        """
        Get list of information about all video Tags for a given project ID.

        :param dataset_id: Project ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Tags.
        :type filters: List[dict], optional
        :return: Information about Tags. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 124976
            tag_infos = api.video.tag.get_list(project_id)
            print(tag_infos)
            # Output: [
            #     [
            #         29098692,
            #         124976,
            #         "number_of_objects",
            #         {
            #             "type": "any_number",
            #             "options": {
            #                 "autoIncrement": false
            #             }
            #         },
            #         "#1F380F",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z"
            #     ],
            #     [
            #         29098693,
            #         124976,
            #         "objects_present",
            #         {
            #             "type": "none",
            #             "options": {}
            #         },
            #         "#F8E71C",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z"
            #     ],
            #     [
            #         29098694,
            #         124976,
            #         "vehicle_colour",
            #         {
            #             "type": "any_string",
            #             "options": {}
            #         },
            #         "#65C0D7",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z"
            #     ],
            #     [
            #         29098695,
            #         124976,
            #         "animal_present",
            #         {
            #             "type": "none",
            #             "options": {}
            #         },
            #         "#872D8B",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z"
            #     ],
            #     [
            #         29098696,
            #         124976,
            #         "animal_age_group",
            #         {
            #             "type": "oneof_string",
            #             "values": [
            #                 "juvenile",
            #                 "adult",
            #                 "senior"
            #             ],
            #             "options": {}
            #         },
            #         "#14902E",
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z"
            #     ]
            # ]
        """
        return self.get_list_all_pages('tags.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})

    def get_name_to_id_map(self, project_id: int) -> Dict[str, int]:
        """
        Get matching the tag name to its ID.

        :param dataset_id: Project ID in Supervisely.
        :type dataset_id: int
        :return: Matching Tag name to it ID in Supervisely
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 124976
            tags_to_ids = api.video.tag.get_name_to_id_map(project_id)
            print(tags_to_ids)
            # Output: {
            #     "number_of_objects": 29098692,
            #     "objects_present": 29098693,
            #     "vehicle_colour": 29098694,
            #     "animal_present": 29098695,
            #     "animal_age_group": 29098696
            # }
        """
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

    def append_to_entity(self, entity_id: int, project_id: int, tags: KeyIndexedCollection, key_id_map: Optional[KeyIdMap] = None) -> List[int]:
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

    def append_to_objects(self, entity_id: int, project_id: int, objects: KeyIndexedCollection, key_id_map: KeyIdMap) -> None:
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

    def append_to_objects_json(self, entity_id: int, tags_json: List[Dict]) -> List[int]:
        if len(tags_json) == 0:
            return []
        response = self._api.post('annotation-objects.tags.bulk.add', {ApiField.ENTITY_ID: entity_id, ApiField.TAGS: tags_json})
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids


