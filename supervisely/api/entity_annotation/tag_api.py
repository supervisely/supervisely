# coding: utf-8

from typing import Any, Dict, List, Optional, Union

from supervisely._utils import batched, take_with_default
from supervisely.annotation.tag_meta import (
    TagValueType,
    validate_frame_range_length_limits,
)
from supervisely.api.module_api import ApiField, ModuleApi
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.imaging.color import random_rgb, rgb2hex
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_settings import LabelingInterface
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap


class TagApi(ModuleApi):
    """Base API module for working with tag metas and appending tags to entities/objects."""

    _entity_id_field = None
    """"""
    _method_bulk_add = None
    """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple TagInfo information about Tag.

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

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

        :returns: NamedTuple name.
        :rtype: str

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                tuple_name = api.video.tag.info_tuple_name()
                print(tuple_name) # TagInfo
        """

        return "TagInfo"

    def get_list(self, project_id: int, filters=None):
        """
        Get list of tags for a given project ID.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param filters: List of parameters to sort output tags. See: https://api.docs.supervisely.com/#tag/Advanced/paths/~1tags.list/get
        :type filters: List[Dict[str, str]], optional
        :returns: List of the tags from the project with given id.
        :rtype: list
        """

        return self.get_list_all_pages(
            "tags.list", {ApiField.PROJECT_ID: project_id, "filter": filters or []}
        )

    def get_name_to_id_map(self, project_id: int):
        """
        Get dictionary with mapping tag name to tag ID for a given project ID.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :returns: Dictionary with mapping tag name to tag id for a given project ID.
        :rtype: dict
        """

        tags_info = self.get_list(project_id)
        return {tag_info.name: tag_info.id for tag_info in tags_info}

    @staticmethod
    def _color_to_hex(color: Union[str, List[int]]) -> str:
        """Accept both a HEX string and an [R, G, B] list, as TagMeta stores RGB."""
        if isinstance(color, str):
            return color
        return rgb2hex(color)

    @staticmethod
    def _frame_range_length_settings(
        min_length: Optional[int],
        max_length: Optional[int],
    ) -> Dict:
        """
        Build the frame range length part of a tag meta ``settings`` payload.

        The server has no separate on/off switch for these limits: a limit is active
        only while its value is non-zero, so an explicit 0 is what disables it. A None
        limit is left out of the payload entirely - on create the server defaults it to
        0 (no limit), on update it keeps the stored value.
        """
        validate_frame_range_length_limits(min_length, max_length)

        settings = {}
        if min_length is not None:
            settings[ApiField.FRAME_RANGE_MIN_LENGTH] = min_length
        if max_length is not None:
            settings[ApiField.FRAME_RANGE_MAX_LENGTH] = max_length
        return settings

    def create_metas(self, project_id: int, tags: List[Dict]) -> List[Dict]:
        """
        Create tag metas (tag definitions) in a project.

        Each item accepts the same keyword arguments as :func:`create_meta`, except
        ``project_id``. Tag names must be unique within the project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags: List of tag meta descriptions.
        :type tags: List[dict]
        :returns: List of created tag metas, each with "id" and "title"
        :rtype: List[dict]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                created = api.video.tag.create_metas(
                    project_id=123,
                    tags=[
                        {"name": "running", "target_type": sly.TagTargetType.FRAME_BASED,
                         "frame_range_min_length": 5, "frame_range_max_length": 30},
                        {"name": "standing", "target_type": sly.TagTargetType.FRAME_BASED},
                    ],
                )
        """
        payload = {
            ApiField.PROJECT_ID: project_id,
            ApiField.TAGS: [self._tag_meta_json(**tag) for tag in tags],
        }
        response = self._api.post("tags.bulk.add", payload)
        return response.json()

    def create_meta(
        self,
        project_id: int,
        name: str,
        value_type: str = TagValueType.NONE,
        possible_values: Optional[List[str]] = None,
        color: Optional[Union[str, List[int]]] = None,
        hotkey: Optional[str] = None,
        applicable_to: Optional[str] = None,
        applicable_class_ids: Optional[List[int]] = None,
        target_type: Optional[str] = None,
        frame_range_min_length: Optional[int] = None,
        frame_range_max_length: Optional[int] = None,
    ) -> Dict:
        """
        Create a single tag meta (tag definition) in a project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Tag name, must be unique within the project.
        :type name: str
        :param value_type: TagValueType: NONE, ANY_STRING, ANY_NUMBER, ONEOF_STRING, DATE.
        :type value_type: str, optional
        :param possible_values: Required for ONEOF_STRING; list of allowed values.
        :type possible_values: List[str], optional
        :param color: Color as a HEX string ("#FF7800") or an [R, G, B] list. Random if not provided.
        :type color: str or List[int], optional
        :param hotkey: Single-character hotkey in the annotation UI.
        :type hotkey: str, optional
        :param applicable_to: TagApplicableTo: ALL, IMAGES_ONLY, OBJECTS_ONLY.
        :type applicable_to: str, optional
        :param applicable_class_ids: Class IDs the tag is restricted to. Only for OBJECTS_ONLY.
                                     Note that this endpoint takes IDs, not class names.
        :type applicable_class_ids: List[int], optional
        :param target_type: TagTargetType: ALL, FRAME_BASED, GLOBAL. Videos and point cloud episodes only.
        :type target_type: str, optional
        :param frame_range_min_length: Minimum length (in frames, inclusive) of a finished
            frame range tag. 0 or None means no limit.
        :type frame_range_min_length: int, optional
        :param frame_range_max_length: Maximum length (in frames, inclusive) of a finished
            frame range tag. 0 or None means no limit.
        :type frame_range_max_length: int, optional
        :raises ValueError: If the frame range limits are negative or min is greater than max.
        :returns: Created tag meta with "id" and "title"
        :rtype: dict

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                # frame range tag that must cover between 5 and 30 frames
                tag = api.video.tag.create_meta(
                    project_id=123,
                    name="running",
                    target_type=sly.TagTargetType.FRAME_BASED,
                    frame_range_min_length=5,
                    frame_range_max_length=30,
                )
        """
        created = self.create_metas(
            project_id,
            [
                {
                    "name": name,
                    "value_type": value_type,
                    "possible_values": possible_values,
                    "color": color,
                    "hotkey": hotkey,
                    "applicable_to": applicable_to,
                    "applicable_class_ids": applicable_class_ids,
                    "target_type": target_type,
                    "frame_range_min_length": frame_range_min_length,
                    "frame_range_max_length": frame_range_max_length,
                }
            ],
        )
        return created[0]

    def _tag_meta_json(
        self,
        name: str,
        value_type: str = TagValueType.NONE,
        possible_values: Optional[List[str]] = None,
        color: Optional[Union[str, List[int]]] = None,
        hotkey: Optional[str] = None,
        applicable_to: Optional[str] = None,
        applicable_class_ids: Optional[List[int]] = None,
        target_type: Optional[str] = None,
        frame_range_min_length: Optional[int] = None,
        frame_range_max_length: Optional[int] = None,
    ) -> Dict:
        """Build a single tag meta item for the "tags.bulk.add" payload."""
        settings = {ApiField.TYPE: value_type}
        if possible_values is not None:
            settings[ApiField.VALUES] = possible_values
        if applicable_to is not None:
            settings[ApiField.APPLICABLE_TYPE] = applicable_to
        if applicable_class_ids is not None:
            settings[ApiField.CLASSES] = applicable_class_ids
        if target_type is not None:
            settings[ApiField.TARGET_TYPE] = target_type
        settings.update(
            self._frame_range_length_settings(frame_range_min_length, frame_range_max_length)
        )

        tag_json = {
            ApiField.TITLE: name,
            ApiField.COLOR: self._color_to_hex(take_with_default(color, random_rgb())),
            ApiField.SETTINGS: settings,
        }
        if hotkey is not None:
            tag_json[ApiField.HOTKEY] = hotkey
        return tag_json

    def update_meta(
        self,
        id: int,
        project_id: Optional[int] = None,
        name: Optional[str] = None,
        color: Optional[Union[str, List[int]]] = None,
        hotkey: Optional[str] = None,
        applicable_to: Optional[str] = None,
        applicable_class_ids: Optional[List[int]] = None,
        target_type: Optional[str] = None,
        frame_range_min_length: Optional[int] = None,
        frame_range_max_length: Optional[int] = None,
    ) -> Dict:
        """
        Update an existing tag meta (tag definition) on the server.

        This is a partial update: only the arguments you pass are changed. Pass 0 to
        drop a frame range length limit - there is no separate on/off switch on the
        server side, a limit is active only while its value is non-zero.

        The endpoint requires "title" and "color" in every request, so pass either both
        of them or ``project_id``, which lets the current values be read from the server.

        :param id: Tag meta ID in Supervisely.
        :type id: int
        :param project_id: Project the tag belongs to. Only needed to look up the current
                           name and color when they are not passed explicitly.
        :type project_id: int, optional
        :param name: New tag name.
        :type name: str, optional
        :param color: New color as a HEX string ("#FF7800") or an [R, G, B] list.
        :type color: str or List[int], optional
        :param hotkey: New single-character hotkey.
        :type hotkey: str, optional
        :param applicable_to: TagApplicableTo: ALL, IMAGES_ONLY, OBJECTS_ONLY.
        :type applicable_to: str, optional
        :param applicable_class_ids: Class IDs the tag is restricted to. Only for OBJECTS_ONLY.
                                     Note that this endpoint takes IDs, not class names.
        :type applicable_class_ids: List[int], optional
        :param target_type: TagTargetType: ALL, FRAME_BASED, GLOBAL. Videos and point cloud episodes only.
        :type target_type: str, optional
        :param frame_range_min_length: Minimum length (in frames, inclusive) of a finished
            frame range tag. 0 removes the limit, None keeps the stored value.
        :type frame_range_min_length: int, optional
        :param frame_range_max_length: Maximum length (in frames, inclusive) of a finished
            frame range tag. 0 removes the limit, None keeps the stored value.
        :type frame_range_max_length: int, optional
        :raises ValueError: If nothing to update is given, the current name and color can
            neither be derived nor looked up, or the frame range limits are negative or
            min is greater than max.
        :returns: Updated tag meta with "id" and "title"
        :rtype: dict

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                # tighten the limits
                api.video.tag.update_meta(
                    id=456, project_id=123, frame_range_min_length=10, frame_range_max_length=20
                )

                # drop the upper limit, keep the lower one
                api.video.tag.update_meta(id=456, project_id=123, frame_range_max_length=0)
        """
        settings = {}
        if applicable_to is not None:
            settings[ApiField.APPLICABLE_TYPE] = applicable_to
        if applicable_class_ids is not None:
            settings[ApiField.CLASSES] = applicable_class_ids
        if target_type is not None:
            settings[ApiField.TARGET_TYPE] = target_type
        settings.update(
            self._frame_range_length_settings(frame_range_min_length, frame_range_max_length)
        )

        if not settings and name is None and color is None and hotkey is None:
            raise ValueError(
                f"To update the tag with ID: {id}, you must specify at least one parameter to "
                "update; all are currently None"
            )

        if name is None or color is None:
            if project_id is None:
                raise ValueError(
                    f"To update the tag with ID: {id}, pass either both name and color, or "
                    "project_id so that the current values can be read from the server"
                )
            stored = next((tag for tag in self.get_list(project_id) if tag.id == id), None)
            if stored is None:
                raise ValueError(f"Tag with ID: {id} is not found in project {project_id}")
            name = take_with_default(name, stored.name)
            color = take_with_default(color, stored.color)

        payload = {
            ApiField.ID: id,
            ApiField.TITLE: name,
            ApiField.COLOR: self._color_to_hex(color),
        }
        if hotkey is not None:
            payload[ApiField.HOTKEY] = hotkey
        if settings:
            payload[ApiField.SETTINGS] = settings

        response = self._api.post("advanced.tags.editInfo", payload)
        return response.json()

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
        :type tags: :class:`~supervisely.collection.key_indexed_collection.KeyIndexedCollection`
        :param key_id_map: KeyIdMap object. See class :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`, optional
        :returns: List of tags IDs
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
        self,
        entity_id: int,
        project_id: int,
        objects: KeyIndexedCollection,
        key_id_map: KeyIdMap,
        is_video_multi_view: bool = False,
    ):
        """
        Add Tags to Annotation Objects for a specific entity (image etc.).

        :param entity_id: ID of the entity in Supervisely to add a tag to its objects
        :type entity_id: int
        :param project_id: Project ID in Supervisely. Uses to get tag name to tag ID mapping.
        :type project_id: int
        :param objects: Collection of annotation objects.
        :type objects: :class:`~supervisely.collection.key_indexed_collection.KeyIndexedCollection`
        :param key_id_map: KeyIdMap object.
        :type key_id_map: :class:`~supervisely.video_annotation.key_id_map.KeyIdMap`
        :param is_video_multi_view: If True, indicates that the entity is a multi-view video.
        :type is_video_multi_view: bool
        :returns: List of tags IDs
        :rtype: list

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

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
        ids = self.append_to_objects_json(entity_id, tags_to_add, project_id, is_video_multi_view)
        KeyIdMap.add_tags_to(key_id_map, tags_keys, ids)
        return ids

    def append_to_objects_json(
        self,
        entity_id: int,
        tags_json: List[Dict],
        project_id: Optional[int] = None,
        is_video_multi_view: bool = False,
    ) -> List[int]:
        """
        Add Tags to Annotation Objects for specific entity (image etc.).

        :param entity_id: ID of the entity in Supervisely to add a tag to its objects
        :type entity_id: int
        :param tags_json: Collection of tags in JSON format
        :type tags_json: dict
        :param project_id: Project ID in Supervisely. Uses to get tag name to tag ID mapping.
                           Not required if `multi_view` is True.
        :type project_id: int, optional
        :param is_video_multi_view: If True, indicates that the entity is a multi-view video.
        :type is_video_multi_view: bool
        :returns: List of tags IDs
        :rtype: list

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                tags_list = [
                    {"tagId": 25926, "objectId": 652959, "value": None},
                    {"tagId": 25927, "objectId": 652959, "value": "v1"},
                    {"tagId": 25927, "objectId": 652958, "value": "v2"},
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
        project_meta = self._api.optimization_context.get("project_meta")

        if isinstance(project_meta, ProjectMeta):
            if project_meta.labeling_interface == LabelingInterface.MULTIVIEW:
                is_video_multi_view = True

        if len(tags_json) == 0:
            return []
        if project_id is not None and not is_video_multi_view:
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
        is_video_multi_view: bool = False,
        entity_id: Optional[int] = None,
    ) -> List[Dict[str, Union[str, int, None]]]:
        """
        Add tags to existing figures/objects.

        - **Images projects**: tags are added to existing *figures* (labels).
          Example element of ``tags_list``: ``{"tagId": 12345, "figureId": 54321, "value": "tag_value"}``.
        - **Video / pointcloud / volume / pointcloud episodes projects**: tags are added to existing *objects*.
          ``frameRange`` is optional and is supported only for video and pointcloud episodes projects.
          Example element of ``tags_list``: ``{"tagId": 12345, "objectId": 54321, "value": "tag_value"}``,
          or ``{"tagId": 12345, "objectId": 54321, "value": "tag_value", "frameRange": [1, 10]}``.

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
        :param is_video_multi_view: If True, indicates that the entity is a multi-view video.
        :type is_video_multi_view: bool
        :param entity_id: ID of the entity in Supervisely to add a tag to its objects.
                          Required if `is_video_multi_view` is True.
        :type entity_id: Optional[int]
        :returns: List of tags infos as dictionaries.
        :rtype: List[Dict[str, Union[str, int, None]]]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                tags_list = [
                    {
                        "tagId": 25926,
                        "figureId": 652959,
                        "value": None,  # optional for tag with type 'None'
                        "frameRange": [1, 10],  # optional (only for video / pointcloud episodes)
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

        project_meta = self._api.optimization_context.get("project_meta")

        if isinstance(project_meta, ProjectMeta):
            if project_meta.labeling_interface == LabelingInterface.MULTIVIEW:
                is_video_multi_view = True

        result = []

        if len(tags_list) == 0:
            return result
        if log_progress:
            progress = tqdm_sly(
                desc="Adding tags to figures",
                total=len(tags_list),
            )
        for batch in batched(tags_list, batch_size):
            if is_video_multi_view:
                if entity_id is None:
                    raise ValueError("entity_id must be provided when is_video_multi_view is True")
                data = {ApiField.ENTITY_ID: entity_id, ApiField.TAGS: batch}
            else:
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
        :returns: List of tags IDs.
        :rtype: List[int]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                tags_list = [
                    {
                        "tagId": 25926,
                        "entityId": 652959,
                        "value": None,  # optional for tag with type 'None'
                        "frameRange": [1, 10],  # optional (only for video)
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
                api.image.tag.add_to_entities_json(project_id=12345, tags_list=tags_list)
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
        is_video_multi_view: bool = False,
        entity_id: Optional[int] = None,
    ) -> List[Dict[str, Union[str, int, None]]]:
        """
        Add a :class:`~supervisely.annotation.tag_collection.TagCollection` to each figure/object.

        - **Images projects**: mapping is ``{figure_id: TagCollection, ...}``
        - **Video / pointcloud / volume / pointcloud episodes projects**: mapping is ``{object_id: TagCollection, ...}``

        All objects must belong to entities of the same project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param tags_map: Dictionary with mapping figure/object ID to tags collection.
        :type tags_map: Dict[int, Any]
        :param batch_size: Number of tags to add in one request.
        :type batch_size: int
        :param log_progress: If True, will display a progress bar.
        :type log_progress: bool
        :param is_video_multi_view: If True, indicates that the entity is a multi-view video.
        :type is_video_multi_view: bool
        :param entity_id: ID of the entity in Supervisely to add a tag to its objects.
                          Required if `is_video_multi_view` is True.
        :type entity_id: Optional[int]
        :returns: List of tags infos as dictionaries.
        :rtype: List[Dict[str, Union[str, int, None]]]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

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
        tag_name_id_map = self.get_name_to_id_map(project_id)

        data = []
        for obj_id, tags in tags_map.items():
            for tag in tags:
                tag_id = tag_name_id_map.get(tag.name)
                if tag_id is None:
                    raise ValueError(f"Tag {tag.name} not found in project {project_id}")

                tag_json = tag.to_json()
                tag_json[ApiField.TAG_ID] = tag_id
                tag_json[OBJ_ID_FIELD] = obj_id
                data.append(tag_json)

        return self.add_to_objects(
            project_id,
            data,
            batch_size,
            log_progress,
            is_video_multi_view=is_video_multi_view,
            entity_id=entity_id,
        )
