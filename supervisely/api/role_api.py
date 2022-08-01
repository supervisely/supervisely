# coding: utf-8
"""list all user roles that are available on private supervisely instance"""

# docs
from __future__ import annotations
from typing import List, Dict, Optional, NamedTuple

from enum import IntEnum
from supervisely.api.module_api import ApiField, ModuleApiBase


class RoleInfo(NamedTuple):
    id: int
    role: str
    created_at: str
    updated_at: str


class RoleApi(ModuleApiBase):
    """
    API for working with Roles. :class:`RoleApi<RoleApi>` object is immutable.

    :param api: API connection to the server
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        roles = api.role.get_list() # api usage example
    """

    class DefaultRole(IntEnum):
        """
        """

        ADMIN = 1
        """"""
        DEVELOPER = 2
        """"""
        ANNOTATOR = 3
        """"""
        VIEWER = 4
        """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple RoleInfo information about Role.

        :Example:

         .. code-block:: python

            RoleInfo(id=71,
                     role='manager',
                     created_at='2019-12-10T14:31:41.878Z',
                     updated_at='2019-12-10T14:31:41.878Z')
        """
        return [ApiField.ID,
                ApiField.ROLE,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **RoleInfo**.
        """
        return 'RoleInfo'

    def get_list(self, filters: Optional[List[Dict[str, str]]] = None) -> List[RoleInfo]:
        """
        List of all roles that are available on private Supervisely instance.

        :param filters: List of params to sort output Roles.
        :type filters: list
        :return: List of all roles with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[RoleInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            roles = api.role.get_list()
        """
        return self.get_list_all_pages('roles.list', {ApiField.FILTER: filters or []})

    def _convert_json_info(self, info: dict, skip_missing=True):
        """
        """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return RoleInfo(**res._asdict())
