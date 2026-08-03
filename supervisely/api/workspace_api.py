# coding: utf-8
"""List and manage Supervisely workspaces."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Union

from supervisely.api.module_api import ApiField, ModuleApi, UpdateableModule
from supervisely.sly_logger import logger


class WorkspaceInfo(NamedTuple):
    """NamedTuple describing a workspace returned by the API."""

    id: int
    name: str
    description: str
    team_id: int
    created_at: str
    updated_at: str


class WorkspaceApi(ModuleApi, UpdateableModule):
    """API for working with workspaces."""

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()
                workspace_info = api.workspace.get_info_by_id(workspace_id)
        """
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    @staticmethod
    def info_sequence():
        """
        Sequence of fields that are returned by the API to represent WorkspaceInfo.

        :Usage Example:

            .. code-block:: python

                WorkspaceInfo(
                    id=15,
                    name='Cars',
                    description='Workspace contains Project with annotated Cars',
                    team_id=8,
                    created_at='2020-04-15T10:50:41.926Z',
                    updated_at='2020-04-15T10:50:41.926Z'
                )
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.TEAM_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Name of the tuple that represents WorkspaceInfo.
        """
        return "WorkspaceInfo"

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[WorkspaceInfo]:
        """
        List of Workspaces in the given Team on the Supervisely instance.

        :param team_id: Team ID in which the Workspaces are located.
        :type team_id: int
        :param filters: List of params to sort output Workspaces.
        :type filters: List[Dict[str, str]], optional
        :returns: List of all Workspaces with information for the given Team.
        :rtype: List[:class:`~supervisely.api.workspace_api.WorkspaceInfo`]

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

                workspace_infos = api.workspace.get_list(8)
                print(workspace_infos)
                # Output: [
                # WorkspaceInfo(id=15,
                #               name='Cars',
                #               description='',
                #               team_id=8,
                #               created_at='2020-04-15T10:50:41.926Z',
                #               updated_at='2020-04-15T10:50:41.926Z'),
                # WorkspaceInfo(id=18,
                #               name='Heart',
                #               description='',
                #               team_id=8,
                #               created_at='2020-05-20T15:01:54.172Z',
                #               updated_at='2020-05-20T15:01:54.172Z'),
                # WorkspaceInfo(id=20,
                #               name='PCD',
                #               description='',
                #               team_id=8,
                #               created_at='2020-06-24T11:51:11.336Z',
                #               updated_at='2020-06-24T11:51:11.336Z')
                # ]

                # Filtered Workspace list
                workspace_infos = api.workspace.get_list(8, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Heart'}])
                print(workspace_infos)
                # Output: [WorkspaceInfo(id=18,
                #                       name='Heart',
                #                       description='',
                #                       team_id=8,
                #                       created_at='2020-05-20T15:01:54.172Z',
                #                       updated_at='2020-05-20T15:01:54.172Z')
                # ]
        """
        return self.get_list_all_pages(
            "workspaces.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters or []},
        )

    def get_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> WorkspaceInfo:
        """
        Get Workspace information by Workspace ID.

        :param id: Workspace ID in Supervisely.
        :type id: int
        :returns: Information about Workspace.
        :rtype: :class:`~supervisely.api.workspace_api.WorkspaceInfo`

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

                workspace_info = api.workspace.get_info_by_id(58)
                print(workspace_info)
                # Output: WorkspaceInfo(id=58,
                #                       name='Test',
                #                       description='',
                #                       team_id=8,
                #                       created_at='2020-11-09T18:21:08.202Z',
                #                       updated_at='2020-11-09T18:21:08.202Z')
        """
        info = self._get_info_by_id(id, "workspaces.info")
        if info is None and raise_error is True:
            raise KeyError(f"Workspace with id={id} not found in your account")
        return info

    def create(
        self,
        team_id: int,
        name: str,
        description: str = "",
        change_name_if_conflict: bool = False,
        hidden: bool = False,
    ) -> WorkspaceInfo:
        """
        Create a new Workspace with the given name in the given Team.

        :param team_id: Team ID in Supervisely where Workspace will be created.
        :type team_id: int
        :param name: Workspace Name.
        :type name: str
        :param description: Workspace description.
        :type description: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool
        :param hidden: Whether the workspace should be hidden or not.
        :type hidden: bool
        :returns: Information about Workspace.
        :rtype: :class:`~supervisely.api.workspace_api.WorkspaceInfo`

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

                new_workspace = api.workspace.create(8, "Vehicle Detection")
                print(new_workspace)
                # Output: WorkspaceInfo(id=274,
                #                       name='Vehicle Detection"',
                #                       description='',
                #                       team_id=8,
                #                       created_at='2021-03-11T12:24:21.773Z',
                #                       updated_at='2021-03-11T12:24:21.773Z')
        """
        effective_name = self._get_effective_new_name(
            parent_id=team_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
        )
        response = self._api.post(
            "workspaces.add",
            {
                ApiField.TEAM_ID: team_id,
                ApiField.NAME: effective_name,
                ApiField.DESCRIPTION: description,
            },
        )
        workspace_info = self._convert_json_info(response.json())
        if hidden:
            try:
                self.change_visibility(workspace_info.id, visible=False)
            except Exception as e:
                logger.error(
                    f"Workspace id={workspace_info.id} was created but could not be set as hidden: {e}. "
                    f"Call change_visibility({workspace_info.id}, visible=False) to retry."
                )

        return workspace_info

    def _get_update_method(self):
        """ """
        return "workspaces.editInfo"

    def archive(self, id: int) -> None:
        """
        Archive Workspace by ID.

        The Workspace is not deleted from the database, it is hidden from the Workspace list and
        can be restored. To delete the Workspace and all its data permanently use
        :func:`remove_permanently`.

        :param id: Workspace ID in Supervisely.
        :type id: int
        :returns: None
        :rtype: None

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

                api.workspace.archive(58)
        """
        self._api.post("workspaces.archive", {ApiField.ID: id})

    def remove_permanently(
        self, ids: Union[int, List], batch_size: int = 50, progress_cb=None
    ) -> List[dict]:
        """
        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        Delete permanently workspaces with given IDs from the Supervisely server.
        Workspaces must be archived with :func:`archive` before calling this method, otherwise the
        server rejects the request. The method is available only for the instance administrator
        (root user), a regular user token is not enough.

        :param ids: IDs of workspaces in Supervisely.
        :type ids: Union[int, List]
        :param batch_size: The number of entities that will be deleted by a single API call. This value must be in the range 1-50 inclusive, if you set a value out of range it will automatically adjust to the boundary values.
        :type batch_size: int, optional
        :param progress_cb: Function for control delete progress.
        :type progress_cb: Callable, optional
        :returns: A list of response content in JSON format for each API call.
        :rtype: List[dict]

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

                workspace_ids = [58, 59]
                for workspace_id in workspace_ids:
                    api.workspace.archive(workspace_id)
                responses = api.workspace.remove_permanently(workspace_ids)
        """
        if batch_size > 50:
            batch_size = 50
        elif batch_size < 1:
            batch_size = 1

        if isinstance(ids, int):
            workspaces = [ids]
        else:
            workspaces = list(ids)

        batches = [workspaces[i : i + batch_size] for i in range(0, len(workspaces), batch_size)]
        responses = []
        for batch in batches:
            response = self._api.post(
                "workspaces.remove.permanently", {ApiField.WORKSPACES_IDS: batch}
            )
            if progress_cb is not None:
                progress_cb(len(batch))
            responses.append(response.json())
        return responses

    def _convert_json_info(self, info: dict, skip_missing=True) -> WorkspaceInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return WorkspaceInfo(**res._asdict())

    def change_visibility(self, id: int, visible: bool):
        """
        Change Workspace visibility by Workspace ID.

        :param id: Workspace ID.
        :type id: int
        :param visible: Visibility status.
        :type visible: bool

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

                api.workspace.change_visibility(58, False)
        """

        response = self._api.post(
            "workspaces.visibility.set",
            {ApiField.ID: id, ApiField.HIDDEN: not visible},
        )
