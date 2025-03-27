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
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

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

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return EntitiesCollectionInfo(**res._asdict())

    def create(self, project_id: int, name: str) -> List[EntitiesCollectionInfo]:
        """
        Creates Entities Collections.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Entities Collection name in Supervisely.
        :type name: str
        :return: List of information about new Entities Collection
        :rtype: :class:`dict`
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
        response = self._api.post("entities-collections.add", data)
        return response.json()  # {"id": 1, "name": "To Annotate"}

    def get_list(
        self,
        project_id: Optional[int] = None,
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

    def _remove_api_method_name(self):
        """ """
        return "entities-collections.remove"

    def _get_update_method(self):
        """ """
        return "entities-collections.editInfo"
