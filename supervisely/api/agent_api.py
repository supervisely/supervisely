# coding: utf-8
"""Create and manage Agents in Supervisely."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleApi, ModuleWithStatus


class AgentInfo(NamedTuple):
    """
    NamedTuple with agent information.
    """

    id: int
    name: str
    token: str
    status: str
    user_id: int
    team_id: int
    capabilities: dict
    created_at: str
    updated_at: str
    is_public: Optional[bool]
    type: Optional[str]
    disabled: Optional[bool]
    version: Optional[str]
    gpu_info: Optional[dict]


class AgentNotFound(Exception):
    """Agent was not found."""

    pass


class AgentNotRunning(Exception):
    """Agent is not running."""

    pass


class AgentApi(ModuleApi, ModuleWithStatus):
    """
    API for working with agents. :class:`~supervisely.api.agent_api.AgentApi` object is immutable.

    :param api: API connection to the server
    :type api: :class:`~supervisely.api.api.Api`

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

            team_id = 8
            agents = api.agent.get_list(team_id)
    """

    class Status(Enum):
        """Agent runtime status values returned by the platform."""

        WAITING = "waiting"
        """"""
        RUNNING = "running"
        """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple AgentInfo information about Agent.

        :Usage Example:

            .. code-block:: python

                AgentInfo(
                    id=1,
                    name="Agent",
                    token="***",
                    status="running",
                    user_id=1,
                    team_id=1,
                    capabilities={},
                )
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TOKEN,
            ApiField.STATUS,
            ApiField.USER_ID,
            ApiField.TEAM_ID,
            ApiField.CAPABILITIES,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.IS_PUBLIC,
            ApiField.TYPE,
            ApiField.DISABLED,
            ApiField.VERSION,
            ApiField.GPU_INFO,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **AgentInfo**.
        """
        return "AgentInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(
        self, team_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        List of all agents in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param filters: List of params to sort output Agents.
        :type filters: List[dict], optional
        :returns: List of agents with information. See :meth:`~supervisely.api.agent_api.AgentApi.info_sequence`.
        :rtype: List[NamedTuple]

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

                team_id = 16087
                agents = api.agent.get_list(team_id)

                filter_agents = api.agent.get_list(team_id, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Gorgeous Chicken' }])
        """
        return self.get_list_all_pages("agents.list", {"teamId": team_id, "filter": filters or []})

    def get_list_available(
        self,
        team_id: int,
        show_public: bool = False,
        show_only_running: bool = True,
        has_gpu: bool = False,
        type: Literal[
            "import",
            "import_agent",
            "export",
            "train",
            "inference",
            "infer_rpc",
            "custom",
            "pipeline",
            "python",
            "app",
            "app_gpu",
            "smarttool",
        ] = None,
        plugin_version_id: int = None,
        version: str = None,
        envs: List[dict] = None,
        min_nvidia_driver_version: str = None,
    ) -> List[AgentInfo]:
        """
        Return list of available agents. Available agents are agents that are not disabled and can be used to run the app.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param show_public: Show public agents.
        :type show_public: bool, optional
        :param show_only_running: Show only running agents.
        :type show_only_running: bool, optional
        :param has_gpu: Show only agents with GPU.
        :type has_gpu: bool, optional
        :param type: Filter by agent task type.
        :type type: Literal["import", "import_agent", "export", "train", "inference", "infer_rpc", "custom", "pipeline", "python", "app", "app_gpu", "smarttool"], optional
        :param plugin_version_id: Filter by agent plugin version ID.
        :type plugin_version_id: int, optional
        :param version: Filter by agent version.
        :type version: str, optional
        :param envs: Filter by agent env variables.
        :type envs: List[dict], optional
        :param min_nvidia_driver_version: Filter by minimum nvidia driver version.
        :type min_nvidia_driver_version: str, optional
        :returns: List of agents with information. See :class:`~supervisely.api.agent_api.AgentInfo`.
        :rtype: List[:class:`~supervisely.api.agent_api.AgentInfo`]

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

                team_id = 350
                available_agents = api.agent.get_list_available(
                    team_id=team_id,
                    show_public=True,
                    show_only_running=False,
                    has_gpu=False,
                    type="app",
                    plugin_version_id=1,
                    version="6.7.39",
                    envs=[{"field": "DOCKER_NET"}],
                    min_nvidia_driver_version="546.33"
                )
        """
        data = {
            "teamId": team_id,
            "withoutPublic": not show_public,
            "anyStatus": not show_only_running,
            "needGPU": has_gpu,
        }

        if type is not None:
            data["type"] = type
        if plugin_version_id is not None:
            data["pluginVersionId"] = plugin_version_id
        if version is not None:
            data["version"] = version
        if envs is not None:
            data["envs"] = envs
        if min_nvidia_driver_version is not None:
            data["minNvidiaDriverVersion"] = min_nvidia_driver_version
        response = self._api.post("agents.available", data)
        infos = response.json()
        agent_infos = []
        for info in infos:
            agent_infos.append(self._convert_json_info(info, True))
        return agent_infos

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Agent information by ID.

        :param id: Agent ID in Supervisely.
        :type id: int
        :returns: Information about an agent. See :meth:`~supervisely.api.agent_api.AgentApi.info_sequence`.
        :rtype: NamedTuple

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

                agent = api.agent.get_info_by_id(7)
        """
        return self._get_info_by_id(id, "agents.info")

    def get_status(self, id: int) -> AgentApi.Status:
        """
        Status object containing status of Agent: waiting or running.

        :param id: Agent ID in Supervisely.
        :type id: int
        :returns: Agent Status
        :rtype: :class:`~supervisely.api.agent_api.AgentApi.Status`

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

                agent = api.agent.get_status(7)
        """
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        """raise_for_status"""
        pass

    def _convert_json_info(self, info: dict, skip_missing=False):
        return super()._convert_json_info(info, True)
