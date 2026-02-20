# coding: utf-8
"""list available classes in supervisely project"""

# docs
from typing import Optional, List, Dict

from supervisely.api.module_api import ModuleApi
from supervisely.api.module_api import ApiField


class ProjectClassApi(ModuleApi):
    """API for working with classes in a project."""

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`
        """
        super().__init__(api)

    @staticmethod
    def info_sequence():
        return [ApiField.ID,
                #ApiField.PROJECT_ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SHAPE,
                ApiField.COLOR,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT
                ]

    @staticmethod
    def info_tuple_name():
        """
        Name of the tuple that represents ProjectClassInfo.
        """
        return 'ProjectClassInfo'

    def get_list(self, project_id: int, filters: Optional[List[Dict[str, str]]] = None) -> list:
        """
        List of Object Classes in the given Project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param filters:
        :type filters: list
        :returns: List of classes.
        :rtype: list
        """
        return self.get_list_all_pages('advanced.object_classes.list',  {ApiField.PROJECT_ID: project_id, "filter": filters or []})
