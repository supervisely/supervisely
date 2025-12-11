# coding: utf-8
"""create or manipulate guides that can be assigned to labeling jobs and labeling queues"""

# docs
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleApiBase


class GuideInfo(NamedTuple):
    """
    Information about a Guide.

    :param id: Guide ID in Supervisely.
    :type id: int
    :param name: Guide name.
    :type name: str
    :param description: Guide description.
    :type description: str
    :param file_path: Path to the guide file (PDF or other).
    :type file_path: str
    :param created_at: Guide creation date.
    :type created_at: str
    :param updated_at: Guide last update date.
    :type updated_at: str
    :param created_by_id: ID of the User who created the Guide.
    :type created_by_id: int
    :param team_id: Team ID where the Guide is located.
    :type team_id: int
    :param video_id: ID of the video associated with the guide (if any).
    :type video_id: Optional[int]
    :param disabled_by: ID of the User who disabled the Guide (if disabled).
    :type disabled_by: Optional[int]
    :param disabled_at: Date when the Guide was disabled (if disabled).
    :type disabled_at: Optional[str]
    """

    id: int
    name: str
    description: str
    file_path: str
    created_at: str
    updated_at: str
    created_by_id: int
    team_id: int
    video_id: Optional[int] = None
    disabled_by: Optional[int] = None
    disabled_at: Optional[str] = None


class GuidesApi(ModuleApiBase):
    """
    API for working with Guides. :class:`GuidesApi<GuidesApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication

        api = sly.Api.from_env()

        # Get list of guides in team
        guides = api.guides.get_list(team_id=123)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple GuideInfo information about Guide.

        :Example:

         .. code-block:: python

            GuideInfo(
                id=1,
                name='How to label objects',
                description='Comprehensive guide on object labeling',
                file_path='/path/to/guide.pdf',
                created_at='2023-01-01T00:00:00.000Z',
                updated_at='2025-11-17T18:21:10.217Z',
                created_by_id=1,
                team_id=1,
                video_id=None,
                disabled_by=None,
                disabled_at=None
            )
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.FILE_PATH,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.CREATED_BY_ID,
            ApiField.TEAM_ID,
            ApiField.VIDEO_ID,
            ApiField.DISABLED_BY,
            ApiField.DISABLED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **GuideInfo**.
        """
        return "GuideInfo"

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[GuideInfo]:
        """
        Get list of Guides in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param filters: List of parameters to filter Guides.
        :type filters: List[Dict[str, str]], optional
        :return: List of information about Guides.
        :rtype: :class:`List[GuideInfo]`
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication

            api = sly.Api.from_env()

            team_id = 123
            guides = api.guides.get_list(team_id)
            print(guides)
            # Output: [
            #     GuideInfo(
            #         id=1,
            #         name='How to label objects',
            #         description='Comprehensive guide on object labeling',
            #         file_path='/path/to/guide.pdf',
            #         created_at='2023-01-01T00:00:00.000Z',
            #         updated_at='2025-11-17T18:21:10.217Z',
            #         created_by_id=1,
            #         team_id=1,
            #         video_id=None,
            #         disabled_by=None,
            #         disabled_at=None
            #     )
            # ]
        """
        return self.get_list_all_pages(
            "guides.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []},
        )

    def get_info_by_id(self, id: int) -> GuideInfo:
        """
        Get Guide information by ID.

        :param id: Guide ID in Supervisely.
        :type id: int
        :return: Information about Guide.
        :rtype: :class:`GuideInfo`
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication

            api = sly.Api.from_env()

            guide_id = 1
            guide_info = api.guides.get_info_by_id(guide_id)
            print(guide_info)
            # Output: GuideInfo(
            #     id=1,
            #     name='How to label objects',
            #     description='Comprehensive guide on object labeling',
            #     file_path='/path/to/guide.pdf',
            #     created_at='2023-01-01T00:00:00.000Z',
            #     updated_at='2025-11-17T18:21:10.217Z',
            #     created_by_id=1,
            #     team_id=1,
            #     video_id=None,
            #     disabled_by=None,
            #     disabled_at=None
            # )
        """
        return self._get_info_by_id(id, "guides.info")
