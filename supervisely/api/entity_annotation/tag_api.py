# coding: utf-8

from typing import Any, Dict, List, Optional, Union

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap


class TagApi(ModuleApi):
    """"""

    _entity_id_field = None
    """"""
    _method_bulk_add = None
    """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple TagInfo information about Tag.

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            info_sequence = api.video.tag.info_sequence()
        """

        return [
            ApiField.ID,
            ApiField.PROJECT_ID,
            ApiField.NAME,
            ApiField.SETTINGS,
            ApiField.COLOR,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of NamedTuple for class.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.video.tag.info_tuple_name()
            print(tuple_name) # TagInfo
        """

        return "TagInfo"

    def get_list(self, project_id: int, filters=None):
        """
        Get list of tags for a given project ID.

        :param project_id: :class:`Dataset<supervisely.project.project.Project>` ID in Supervisely.
        :type project_id: int
        :param filters: List of parameters to sort output tags. See: https://api.docs.supervisely.com/#tag/Advanced/paths/~1tags.list/get
        :type filters: List[Dict[str, str]], optional
        :return: List of the tags from the project with given id.
        :rtype: list
        """

        return self.get_list_all_pages(
            "tags.list", {ApiField.PROJECT_ID: project_id, "filter": filters or []}
        )

    def get_name_to_id_map(self, project_id: int):
        """
        Get dictionary with mapping tag name to tag ID for a given project ID.

        :param project_id: :class:`Dataset<supervisely.project.project.Project>` ID in Supervisely.
        :type project_id: int
        :return: Dictionary with mapping tag name to tag id for a given project ID.
        :rtype: dict
        """

        tags_info = self.get_list(project_id)
        return {tag_info.name: tag_info.id for tag_info in tags_info}

    def _tags_to_json(self, tags: KeyIndexedCollection, tag_name_id_map=None, project_id=None):
        """"""
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

    def append_to_entity(
        self,
        entity_id: int,
        project_id: int,
        tags: KeyIndexedCollection,
        key_id_map: KeyIdMap = None,
    ):
        """
        Add tags to entity in project with given ID.

        :param entity_id: ID of the entity in Supervisely to add a tag to
        :type entity_id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags: Collection of tags
        :type tags: KeyIndexedCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: List of tags IDs
        :rtype: list
        """

        if len(tags) == 0:
            return []
        tags_json, tags_keys = self._tags_to_json(tags, project_id=project_id)
        ids = self._append_json(entity_id, tags_json)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)
        return ids

    def _append_json(self, entity_id, tags_json):
        """"""
        if self._method_bulk_add is None:
            raise RuntimeError("self._method_bulk_add is not defined in child class")
        if self._entity_id_field is None:
            raise RuntimeError("self._entity_id_field is not defined in child class")

        if len(tags_json) == 0:
            return []
        response = self._api.post(
            self._method_bulk_add, {self._entity_id_field: entity_id, ApiField.TAGS: tags_json}
        )
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids

    def append_to_objects(
        self, entity_id: int, project_id: int, objects: KeyIndexedCollection, key_id_map: KeyIdMap
    ):
        """
        Add Tags to Annotation Objects for a specific entity (image etc.).

        :param entity_id: ID of the entity in Supervisely to add a tag to its objects
        :type entity_id: int
        :param project_id: Project ID in Supervisely. Uses to get tag name to tag ID mapping.
        :type project_id: int
        :param objects: Collection of Annotation Objects.
        :type objects: KeyIndexedCollection
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: List of tags IDs
        :rtype: list
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_id = 19373170
            img_info = api.
        """

        tag_name_id_map = self.get_name_to_id_map(project_id)

        tags_to_add = []
        tags_keys = []
        for object in objects:
            obj_id = key_id_map.get_object_id(object.key())
            if obj_id is None:
                raise RuntimeError(
                    "Can not add tags to object: OBJECT_ID not found for key {}".format(
                        object.key()
                    )
                )
            tags_json, cur_tags_keys = self._tags_to_json(
                object.tags, tag_name_id_map=tag_name_id_map
            )
            for tag in tags_json:
                tag[ApiField.OBJECT_ID] = obj_id
                tags_to_add.append(tag)
            tags_keys.extend(cur_tags_keys)

        if len(tags_keys) != len(tags_to_add):
            raise RuntimeError("SDK error: len(tags_keys) != len(tags_to_add)")
        if len(tags_keys) == 0:
            return
        ids = self.append_to_objects_json(entity_id, tags_to_add, project_id)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)
        return ids

    def append_to_objects_json(
        self, entity_id: int, tags_json: List[Dict], project_id: Optional[int] = None
    ) -> List[int]:
        """
        Add Tags to Annotation Objects for specific entity (image etc.).

        :param entity_id: ID of the entity in Supervisely to add a tag to its objects
        :type entity_id: int
        :param tags_json: Collection of tags in JSON format
        :type tags_json: dict
        :return: List of tags IDs
        :rtype: list

        :Usage example:

             .. code-block:: python

                import supervisely as sly

                api = sly.Api(server_address, token)

                tags_list = [
                    {
                        "tagId": 25926,
                        "objectId": 652959,
                        "value": None
                    },
                    {
                        "tagId": 25927,
                        "objectId": 652959,
                        "value": "v1"
                    },
                    {
                        "tagId": 25927,
                        "objectId": 652958,
                        "value": "v2"
                    }
                ]
                response = api.video.tag.append_to_objects_json(12345, tags_list)

                print(response)
                # Output:
                #    [
                #        80421101,
                #        80421102,
                #        80421103
                #    ]
        """

        if len(tags_json) == 0:
            return []
        if project_id is not None:
            json_data = {ApiField.PROJECT_ID: project_id, ApiField.TAGS: tags_json}
        else:
            json_data = {ApiField.ENTITY_ID: entity_id, ApiField.TAGS: tags_json}
        response = self._api.post("annotation-objects.tags.bulk.add", json_data)
        ids = [obj[ApiField.ID] for obj in response.json()]
        return ids

    def add_to_objects(
        self,
        project_id: int,
        tags_list: List[dict],
        batch_size: int = 100,
        log_progress: bool = False,
        progress: Optional[tqdm_sly] = None,
    ) -> List[Dict[str, Union[str, int, None]]]:
        """
        For images project:
            Add Tags to existing Annotation Figures (labels).
            The `tags_list` example:
                [{"tagId": 12345, "figureId": 54321, "value": "tag_value"}, ...].
        For video, pointcloud, volume and pointcloud episodes projects:
            Add Tags to existing Annotation Objects.
            The `frameRange` field is optional and is supported only for video and pointcloud episodes projects.
            The `tags_list`` example:
                [{"tagId": 12345, "objectId": 54321, "value": "tag_value"}, ...].
            or with frameRange:
                [{"tagId": 12345, "objectId": 54321, "value": "tag_value", "frameRange": [1, 10]}, ...].

        All objects must belong to entities of the same project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags_list: List of tag object infos as dictionaries.
        :type tags_list: List[dict]
        :param batch_size: Number of tags to add in one request.
        :type batch_size: int
        :param log_progress: If True, will display a progress bar.
        :type log_progress: bool
        :param progress: Progress bar object to display progress.
        :type progress: Optional[tqdm_sly]
        :return: List of tags infos as dictionaries.
        :rtype: List[Dict[str, Union[str, int, None]]]

        Usage example:
        .. code-block:: python

            import supervisely as sly

            api = sly.Api(server_address, token)

            tag_list = [
                {
                    "tagId": 25926,
                    "figureId": 652959,
                    "value": None # optional for tag with type 'None'
                    "frameRange": [1, 10] # optional (supported only for video and pointcloud episodes projects)
                },
                {
                    "tagId": 25927,
                    "figureId": 652959,
                    "value": "v1"
                },
                {
                    "tagId": 25927,
                    "figureId": 652958,
                    "value": "v2",
                }
            ]
            response = api.image.tag.add_to_figures(12345, tag_list)

            print(response)
            # Output:
            #    [
            #        {
            #            "id": 80421101,
            #            "tagId": 25926,
            #            "figureId": 652959,
            #            "value": None
            #        },
            #        {
            #            "id": 80421102,
            #            "tagId": 25927,
            #            "figureId": 652959,
            #            "value": "v1"
            #        },
            #        {
            #            "id": 80421103,
            #            "tagId": 25927,
            #            "figureId": 652958,
            #            "value": "v2"
            #        }
            #    ]
        """

        if progress is not None:
            log_progress = False

        result = []

        if len(tags_list) == 0:
            return result
        if log_progress:
            progress = tqdm_sly(
                desc="Adding tags to figures",
                total=len(tags_list),
            )
        for batch in batched(tags_list, batch_size):
            data = {ApiField.PROJECT_ID: project_id, ApiField.TAGS: batch}
            if type(self) is TagApi:
                response = self._api.post("figures.tags.bulk.add", data)
            else:
                response = self._api.post("annotation-objects.tags.bulk.add", data)
            result.extend(response.json())
            if progress is not None:
                progress.update(len(batch))
        return result

    def add_to_entities_json(
        self,
        project_id: int,
        tags_list: List[Dict[str, Union[str, int, None]]],
        batch_size: int = 100,
        log_progress: bool = False,
    ) -> List[int]:
        """
        Bulk add tags to entities (images, videos, pointclouds, volumes) in a project.
        Not supported for pointcloud episodes projects.
        All entities must belong to the same project.
        The `frameRange` field in a tag object within the tags list is optional and is supported only for video projects.

        The `tags_list` example:
            [{"tagId": 12345, "entityId": 54321, "value": "tag_value"}, ...].
        or with frameRange:
            [{"tagId": 12345, "entityId": 54321, "value": "tag_value", "frameRange": [1, 10]}, ...].

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags_list: List of tag object infos as dictionaries
                          (e.g. {"tagId": 12345, "entityId": 54321, "value": "tag_value"}).
        :param batch_size: Number of tags to add in one request.
        :type batch_size: int
        :param log_progress: If True, will display a progress bar.
        :type log_progress: bool
        :return: List of tags IDs.
        :rtype: List[int]

        Usage example:
        .. code-block:: python

            import supervisely as sly

            api = sly.Api(server_address, token)

            tag_list = [
                {
                    "tagId": 25926,
                    "entityId": 652959,
                    "value": None # optional for tag with type 'None'
                    "frameRange": [1, 10] # optional (supported only for video projects)
                },
                {
                    "tagId": 25927,
                    "entityId": 652959,
                    "value": "v1"
                },
                {
                    "tagId": 25927,
                    "entityId": 652958,
                    "value": "v2"
                }
            ]
            api.image.tag.add_to_entities_json(project_id=12345, tag_list=tag_list)
        """

        result = []

        if len(tags_list) == 0:
            return result

        if log_progress:
            ds_progress = tqdm_sly(desc="Adding tags to entities", total=len(tags_list))

        for batch in batched(tags_list, batch_size):
            data = {ApiField.PROJECT_ID: project_id, ApiField.TAGS: batch}
            response = self._api.post("tags.entities.bulk.add", data)
            result.extend([obj[ApiField.ID] for obj in response.json()])
            if log_progress:
                ds_progress.update(len(batch))

        return result

    def add_tags_collection_to_objects(
        self,
        project_id: int,
        tags_map: Dict[int, Any],
        batch_size: int = 100,
        log_progress: bool = False,
    ) -> List[Dict[str, Union[str, int, None]]]:
        """
        For images project:
            Add Tags to existing Annotation Figures (labels).
            The `tags_map` example: {figure_id_1: TagCollection, ...}.
        For video, pointcloud, volume and pointcloud episodes projects:
            Add Tags to existing Annotation Objects.
            The `frameRange` field is optional and is supported only for video and pointcloud episodes projects.
            The `tags_map` example: {object_id_1: TagCollection, ...}.

        All objects must belong to entities of the same project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags_map: Dictionary with mapping figure/object ID to tags collection.
        :type tags_map: Dict[int, Any]
        :param batch_size: Number of tags to add in one request.
        :type batch_size: int
        :param log_progress: If True, will display a progress bar.
        :type log_progress: bool
        :return: List of tags infos as dictionaries.
        :rtype: List[Dit[str, Union[str, int, None]]]

        Usage example:
        .. code-block:: python

            import supervisely as sly

            api = sly.Api(server_address, token)

            project_id = 12345

            tag_meta = sly.TagMeta("tag_name", sly.TagValueType.ANY_STRING)
            meta = sly.ProjectMeta(tag_metas=[tag_meta])
            meta = sly.ProjectMeta.from_json(api.project.update_meta(project_id, meta))
            tag_meta = meta.get_tag_meta("tag_name")

            # for images project:
            tag_map = {
                652959: sly.TagCollection([sly.Tag(tag_meta, value="v1"), sly.Tag(tag_meta, value="v2"), ...]),
                652958: sly.TagCollection([sly.Tag(tag_meta, value="v3"), sly.Tag(tag_meta, value="v4"), ...]),
                ...
            }
            api.image.tag.add_tags_to_objects(project_id, tag_map)

            # for videos projects (frameRange is optional):
            tag_map = {
                652959: sly.VideoTagCollection([sly.VideoTag(tag_meta, value="v1", frameRange=[1, 10]), ...]),
                652958: sly.VideoTagCollection([sly.VideoTag(tag_meta, value="v2", frameRange=[4, 12]), ...]),
                ...
            }
            api.video.tag.add_to_objects_json_batch(project_id, tag_map)
        """

        OBJ_ID_FIELD = ApiField.FIGURE_ID if type(self) is TagApi else ApiField.OBJECT_ID

        data = []
        for obj_id, tags in tags_map.items():
            for tag in tags:

                if tag.meta.sly_id is None:
                    raise ValueError(f"Tag {tag.name} meta has no sly_id")

                data.append(
                    {
                        ApiField.TAG_ID: tag.meta.sly_id,
                        OBJ_ID_FIELD: obj_id,
                        **tag.to_json()
                    }
                )

        return self.add_to_objects(project_id, data, batch_size, log_progress)
