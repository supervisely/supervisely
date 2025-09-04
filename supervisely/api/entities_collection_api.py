# coding: utf-8
"""create or manipulate already existing Entities Collection in Supervisely"""

# docs
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union

import requests

from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
)
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.api.image_api import ImageInfo


class CollectionType:
    DEFAULT = "default"
    AI_SEARCH = "aiSearch"
    ALL = "all"


class CollectionTypeFilter:
    AI_SEARCH = "entities_ai_search_collection"
    DEFAULT = "entities_collection"


class AiSearchThresholdDirection:
    ABOVE = "above"
    BELOW = "below"


@dataclass
class CollectionItem:
    """
    Collection item with meta information.

    :param entity_id: Supervisely ID of the item.
    :type entity_id: int
    :param meta: Meta information about the item. Optional, defaults to None.
    :type meta: :class:`CollectionItem.Meta`, optional
    """

    @dataclass
    class Meta:
        """
        Meta information about the item.

        :param score: Score value of the item that indicates search relevance.
        :type score: float
        """

        score: Optional[float] = 0.0

        def to_json(self) -> dict:
            """
            Convert meta information to a JSON-compatible dictionary.

            :return: Dictionary with meta information.
            :rtype: dict
            """
            return {ApiField.SCORE: self.score}

        @classmethod
        def from_json(cls, data: dict) -> "CollectionItem.Meta":
            """
            Create Meta from a JSON-compatible dictionary.

            :param data: Dictionary with meta information.
            :type data: dict
            :return: Meta object.
            :rtype: CollectionItem.Meta
            """
            return cls(score=data.get(ApiField.SCORE, 0.0))

    entity_id: int
    meta: Optional[Meta] = None

    def to_json(self) -> dict:
        """
        Convert collection item to a JSON-compatible dictionary.

        :return: Dictionary with collection item data.
        :rtype: dict
        """
        result = {ApiField.ENTITY_ID: self.entity_id}
        if self.meta is not None:
            result[ApiField.META] = self.meta.to_json()
        return result

    @classmethod
    def from_json(cls, data: dict) -> "CollectionItem":
        """
        Create CollectionItem from a JSON-compatible dictionary.

        :param data: Dictionary with collection item data.
        :type data: dict
        :return: CollectionItem object.
        :rtype: CollectionItem
        """
        meta_data = data.get(ApiField.META)
        meta = cls.Meta.from_json(meta_data) if meta_data is not None else None
        return cls(entity_id=data[ApiField.ENTITY_ID], meta=meta)


class EntitiesCollectionInfo(NamedTuple):
    """
    Object with entitites collection parameters from Supervisely.

    :Example:

     .. code-block:: python

        EntitiesCollectionInfo(
            id=1,
            team_id=2,
            project_id=3,
            name="collection_name",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-02T00:00:00Z",
            description="This is a collection",
            options={"key": "value"},
            type="default",
            ai_search_key="search_key"
        )
    """

    #: ID of the collection.
    id: int

    #: Name of the collection.
    name: str

    #: ID of the team.
    team_id: int

    #: ID of the project.
    project_id: int

    #: Date and time when the collection was created.
    created_at: str

    #: Date and time when the collection was last updated.
    updated_at: str

    #: Description of the collection.
    description: Optional[str] = None

    #: Additional options for the collection.
    options: Dict[str, Union[str, int, bool]] = None

    #: Type of the collection.
    type: str = CollectionType.DEFAULT

    #: AI search key for the collection.
    ai_search_key: Optional[str] = None


class EntitiesCollectionApi(UpdateableModule, RemoveableModuleApi):
    """
    API for working with Entities Collection. :class:`EntitiesCollectionApi<EntitiesCollectionApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        collection = api.entities_collection.get_list(9) # api usage example
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple EntitiesCollectionInfo information about Entities Collection.

        :Example:

         .. code-block:: python

             EntitiesCollectionInfo(
                id=2,
                name='Enitites Collections #1',
                team_id=4,
                project_id=58,
                description='',
                created_at='2020-04-08T15:10:12.618Z',
                updated_at='2020-04-08T15:10:19.833Z',
            )
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TEAM_ID,
            ApiField.PROJECT_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.DESCRIPTION,
            ApiField.OPTIONS,
            ApiField.TYPE,
            ApiField.AI_SEARCH_KEY,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`EntitiesCollectionInfo` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        """
        return "EntitiesCollectionInfo"

    def __init__(self, api: Api):
        ModuleApi.__init__(self, api)
        self._api = api

    def _convert_json_info(self, info: dict, skip_missing=True) -> EntitiesCollectionInfo:
        """
        Differs from the original method by using skip_missing equal to True by default.
        Also unpacks 'meta' field to top level for fields in info_sequence.
        """

        def _get_value(dict, field_name, skip_missing):
            if skip_missing is True:
                return dict.get(field_name, None)
            else:
                return dict[field_name]

        if info is None:
            return None
        else:
            field_values = []
            meta = info.get("meta", {})

            for field_name in self.info_sequence():
                if type(field_name) is str:
                    # Try to get from meta first if the field exists there
                    if field_name in meta:
                        field_values.append(meta.get(field_name))
                    else:
                        field_values.append(_get_value(info, field_name, skip_missing))
                elif type(field_name) is tuple:
                    value = None
                    for sub_name in field_name[0]:
                        if value is None:
                            value = _get_value(info, sub_name, skip_missing)
                        else:
                            value = _get_value(value, sub_name, skip_missing)
                    field_values.append(value)
                else:
                    raise RuntimeError("Can not parse field {!r}".format(field_name))
            return self.InfoType(*field_values)

    def create(
        self,
        project_id: int,
        name: str,
        description: Optional[str] = None,
        type: str = CollectionType.DEFAULT,
        ai_search_key: Optional[str] = None,
    ) -> EntitiesCollectionInfo:
        """
        Creates Entities Collections.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Entities Collection name in Supervisely.
        :type name: str
        :param description: Entities Collection description.
        :type description: str
        :param type: Type of the collection. Defaults to "default".
        :type type: str
        :param ai_search_key: AI search key for the collection. Defaults to None.
        :type ai_search_key: Optional[str]
        :return: Information about new Entities Collection
        :rtype: :class:`EntitiesCollectionInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 123
            name = "Chihuahuas"
            description = "Collection of Chihuahuas"
            type = CollectionType.AI_SEARCH
            ai_search_key = "0ed6a5256433bbe32822949d563d476a"

            new_collection = api.entities_collection.create(project_id, name, description, type, ai_search_key)
            print(new_collection)
        """
        method = "entities-collections.add"
        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.NAME: name,
            ApiField.TYPE: type,
        }
        if description is not None:
            data[ApiField.DESCRIPTION] = description
        if ai_search_key is not None:
            if type != CollectionType.AI_SEARCH:
                raise ValueError(
                    "ai_search_key is only available for creation AI Search collection type."
                )
            if ApiField.META not in data:
                data[ApiField.META] = {}
            data[ApiField.META][ApiField.AI_SEARCH_KEY] = ai_search_key
        elif type == CollectionType.AI_SEARCH:
            raise ValueError("ai_search_key is required for AI Search collection type.")
        response = self._api.post(method, data)
        return self._convert_json_info(response.json())

    def remove(self, id: int, force: bool = False):
        """
        Remove Entites Collection with the specified ID from the Supervisely server.

        If `force` is set to True, the collection will be removed permanently.
        If `force` is set to False, the collection will be disabled instead of removed.

        :param id: Entites Collection ID in Supervisely
        :type id: int
        :param force: If True, the collection will be removed permanently. Defaults to False.
        :type force: bool
        :return: None
        """
        self._api.post(
            self._remove_api_method_name(), {ApiField.ID: id, ApiField.HARD_DELETE: force}
        )

    def get_list(
        self,
        project_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        with_meta: bool = False,
        collection_type: CollectionType = CollectionType.DEFAULT,
    ) -> List[EntitiesCollectionInfo]:
        """
        Get list of information about Entities Collection for the given project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param filters: List of filters to apply to the request. Optional, defaults to None.
        :type filters: List[Dict[str, str]], optional
        :param with_meta: If True, includes meta information in the response. Defaults to False.
        :type with_meta: bool, optional
        :param collection_type: Type of the collection.
                    Defaults to CollectionType.DEFAULT.

                    Available types are:
                     - CollectionType.DEFAULT
                     - CollectionType.AI_SEARCH
                     - CollectionType.ALL
        :type collection_type: CollectionType
        :return: List of information about Entities Collections.
        :rtype: :class:`List[EntitiesCollectionInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            collections = api.entities_collection.get_list(4)
        """
        method = "entities-collections.list"

        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.FILTER: filters or [],
            ApiField.TYPE: collection_type,
        }
        if with_meta:
            data.update({ApiField.EXTRA_FIELDS: [ApiField.META]})
        return self.get_list_all_pages(method, data)

    def get_info_by_id(self, id: int, with_meta: bool = False) -> EntitiesCollectionInfo:
        """
        Get information about Entities Collection with given ID.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param with_meta: If True, includes meta information in the response. Defaults to False.
        :type with_meta: bool, optional
        :return: Information about Entities Collection.
        :rtype: :class:`EntitiesCollectionInfo`
        :Usage example:

            .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            collection = api.entities_collection.get_info_by_id(2)
            print(collection)
            # Output:
            # {
            #     "id": 1,
            #     "teamId": 1,
            #     "projectId": 1,
            #     "name": "ds",
            #     "description": "",
            #     "createdAt": "2018-08-21T14:25:56.140Z",
            #     "updatedAt": "2018-08-21T14:25:56.140Z"
            # }
        """
        method = "entities-collections.info"
        if with_meta:
            extra_fields = [ApiField.META]
        else:
            extra_fields = None
        return self._get_info_by_id(id, method, extra_fields)

    def get_info_by_ai_search_key(
        self, project_id: int, ai_search_key: str
    ) -> Optional[EntitiesCollectionInfo]:
        """
        Get information about Entities Collection of type `CollectionType.AI_SEARCH` with given AI search key.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param ai_search_key: AI search key for the collection.
        :type  ai_search_key: str
        :return: Information about Entities Collection.
        :rtype: :class:`EntitiesCollectionInfo`
        :Usage example:
         .. code-block:: python
            import supervisely as sly
            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()
            project_id = 123
            ai_search_key = "0ed6a5256433bbe32822949d563d476a"

            # Get collection by AI search key
            collection = api.entities_collection.get_info_by_ai_search_key(project_id, ai_search_key)
        """
        collections = self.get_list(
            project_id=project_id,
            with_meta=True,
            collection_type=CollectionType.AI_SEARCH,
        )
        return next(
            (collection for collection in collections if collection.ai_search_key == ai_search_key),
            None,
        )

    def _get_response_by_id(self, id, method, id_field, extra_fields=None):
        """Differs from the original method by using extra_fields."""
        try:
            data = {id_field: id}
            if extra_fields is not None:
                data.update({ApiField.EXTRA_FIELDS: extra_fields})
            return self._api.post(method, data)
        except requests.exceptions.HTTPError as error:
            if error.response.status_code == 404:
                return None
            else:
                raise error

    def _get_info_by_id(self, id, method, extra_fields=None):
        """_get_info_by_id"""
        response = self._get_response_by_id(
            id, method, id_field=ApiField.ID, extra_fields=extra_fields
        )
        return self._convert_json_info(response.json()) if (response is not None) else None

    def _get_update_method(self):
        """ """
        return "entities-collections.editInfo"

    def _remove_api_method_name(self):
        """ """
        return "entities-collections.remove"

    def add_items(self, id: int, items: List[Union[int, CollectionItem]]) -> List[Dict[str, int]]:
        """
        Add items to Entities Collection.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param items: List of items to add to the collection. Could be a list of entity IDs (int) or CollectionItem objects.
        :type items: List[Union[int, CollectionItem]]
        :return: List of added items with their IDs and creation timestamps.
        :rtype: List[Dict[str, int]]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.entities_collection_api import CollectionItem

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            collection_id = 2
            item_ids = [525, 526]
            items = [CollectionItem(entity_id=item_id) for item_id in item_ids]
            new_items = api.entities_collection.add_items(collection_id, items)
            print(new_items)
            # Output: [
            #   {"id": 1, "entityId": 525, 'createdAt': '2025-04-10T08:49:41.852Z'},
            #   {"id": 2, "entityId": 526, 'createdAt': '2025-04-10T08:49:41.852Z'}
            ]
        """
        if all(isinstance(item, int) for item in items):
            data = {ApiField.COLLECTION_ID: id, ApiField.ENTITY_IDS: items}
        elif all(isinstance(item, CollectionItem) for item in items):
            items = [item.to_json() for item in items]
            data = {ApiField.COLLECTION_ID: id, ApiField.ENTITY_ITEMS: items}
        else:
            raise ValueError(
                "Items list must contain only integers or only CollectionItem instances, not a mix."
            )

        response = self._api.post("entities-collections.items.bulk.add", data)
        response = response.json()
        if len(response["missing"]) > 0:
            raise RuntimeError(
                f"Failed to add items to Entities Collection. IDs: {response['missing']}. "
            )
        return response["items"]

    def get_items(
        self,
        collection_id: int,
        collection_type: CollectionTypeFilter,
        project_id: Optional[int] = None,
        ai_search_threshold: Optional[float] = None,
        ai_search_threshold_direction: AiSearchThresholdDirection = AiSearchThresholdDirection.ABOVE,
    ) -> List[ImageInfo]:
        """
        Get items from Entities Collection.

        :param collection_id: Entities Collection ID in Supervisely.
        :type collection_id: int
        :param collection_type: Type of the collection. Can be CollectionTypeFilter.AI_SEARCH or CollectionTypeFilter.DEFAULT.
        :type collection_type: CollectionTypeFilter
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param ai_search_threshold: AI search threshold for filtering items. Optional, defaults to None.
        :type ai_search_threshold: float, optional
        :param ai_search_threshold_direction: Direction for the AI search threshold. Optional, defaults to 'above'.
        :type ai_search_threshold_direction: str
        :return: List of ImageInfo objects.
        :rtype: List[ImageInfo]
        :raises RuntimeError: If Entities Collection with given ID not found.
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            collection_id = 2
            project_id = 4
            item_ids = api.entities_collection.get_items(collection_id, project_id)
            print(item_ids)
        """
        if project_id is None:
            info = self.get_info_by_id(collection_id)
            if info is None:
                raise RuntimeError(f"Entities Collection with id={collection_id} not found.")
            project_id = info.project_id

        if collection_type == CollectionTypeFilter.AI_SEARCH:
            return self._api.image.get_list(
                project_id=project_id,
                ai_search_collection_id=collection_id,
                extra_fields=[ApiField.EMBEDDINGS_UPDATED_AT],
                ai_search_threshold=ai_search_threshold,
                ai_search_threshold_direction=ai_search_threshold_direction,
            )
        else:
            return self._api.image.get_list(
                project_id=project_id,
                entities_collection_id=collection_id,
            )

    def remove_items(self, id: int, items: List[int]) -> List[Dict[str, int]]:
        """
        Remove items from Entities Collection.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param items: List of entity IDs in Supervisely, e.g. image IDs etc.
        :type items: List[int]
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            api = sly.Api.from_env()

            collection_id = 2
            item_ids = [525, 526, 527]
            removed_items = api.entities_collection.remove_items(collection_id, item_ids)
            # print(removed_items)
            # Output: [{"id": 1, "entityId": 525}, {"id": 2, "entityId": 526}]
        """
        method = "entities-collections.items.bulk.remove"

        data = {ApiField.COLLECTION_ID: id, ApiField.ENTITY_IDS: items}
        response = self._api.post(method, data)
        response = response.json()
        if len(response["missing"]) > 0:
            logger.warning(
                f"Failed to remove items from Entities Collection. IDs: {response['missing']}. "
            )
        return response["items"]
