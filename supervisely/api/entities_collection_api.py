# coding: utf-8
"""create or manipulate already existing Entities Collection in Supervisely"""

# docs
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
)
from supervisely.sly_logger import logger


class EntitiesCollectionInfo(NamedTuple):
    id: int
    name: str
    team_id: int
    project_id: int
    description: str
    created_at: str
    updated_at: str


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
            ApiField.DESCRIPTION,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **EntitiesCollectionInfo**.
        """
        return "EntitiesCollectionInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)

    def create(
        self, project_id: int, name: str, description: Optional[str] = None
    ) -> EntitiesCollectionInfo:
        """
        Creates Entities Collections.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Entities Collection name in Supervisely.
        :type name: str
        :param description: Entities Collection description.
        :type description: str
        :return: Information about new Entities Collection
        :rtype: :class:`EntitiesCollectionInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_name = 'Collection #1'
            project_id = 602
            new_collection = api.entities_collection.create(name, project_id)
            print(new_collection)
        """

        data = {ApiField.NAME: name, ApiField.PROJECT_ID: project_id}
        if description is not None:
            data[ApiField.DESCRIPTION] = description
        response = self._api.post("entities-collections.add", data)
        return self._convert_json_info(response.json())

    def get_list(
        self,
        project_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
    ) -> List[EntitiesCollectionInfo]:
        """
        Get list of information about Entities Collection for the given project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
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
        if filters is None:
            filters = []
        return self.get_list_all_pages(
            "entities-collections.list",
            {ApiField.PROJECT_ID: project_id, ApiField.FILTER: filters},
        )

    def get_info_by_id(self, id: int) -> EntitiesCollectionInfo:
        """
        Get information about Entities Collection with given ID.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
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
        return self._get_info_by_id(id, "entities-collections.info")

    def _get_update_method(self):
        """ """
        return "entities-collections.editInfo"

    def _remove_api_method_name(self):
        """ """
        return "entities-collections.remove"

    def _get_update_method(self):
        """ """
        return "entities-collections.editInfo"

    def add_items(self, id: int, items: List[int]) -> List[Dict[str, int]]:
        """
        Add items to Entities Collection.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param items: List of item IDs in Supervisely.
        :type items: List[int]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            collection_id = 2
            item_ids = [525, 526]
            new_items = api.entities_collection.add_items(collection_id, item_ids)
            print(new_items)
            # Output: [
            #   {"id": 1, "entityId": 525, 'createdAt': '2025-04-10T08:49:41.852Z'},
            #   {"id": 2, "entityId": 526, 'createdAt': '2025-04-10T08:49:41.852Z'}
            ]
        """
        data = {ApiField.COLLECTION_ID: id, ApiField.ENTITY_IDS: items}
        response = self._api.post("entities-collections.items.bulk.add", data)
        response = response.json()
        if len(response["missing"]) > 0:
            raise RuntimeError(
                f"Failed to add items to Entities Collection. IDs: {response['missing']}. "
            )
        return response["items"]

    def get_items(self, id: int, project_id: Optional[int] = None) -> List[int]:
        """
        Get items from Entities Collection.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :return: List of item IDs in Supervisely.
        :rtype: List[int]
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
            info = self.get_info_by_id(id)
            if info is None:
                raise RuntimeError(f"Entities Collection with id={id} not found.")
            project_id = info.project_id

        return self._api.image.get_list(project_id=project_id, entities_collection_id=id)

    def remove_items(self, id: int, items: List[int]) -> List[Dict[str, int]]:
        """
        Remove items from Entities Collection.

        :param id: Entities Collection ID in Supervisely.
        :type id: int
        :param items: List of item IDs in Supervisely.
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
        data = {ApiField.COLLECTION_ID: id, ApiField.ENTITY_IDS: items}
        response = self._api.post("entities-collections.items.bulk.remove", data)
        response = response.json()
        if len(response["missing"]) > 0:
            logger.warning(
                f"Failed to remove items from Entities Collection. IDs: {response['missing']}. "
            )
        return response["items"]
