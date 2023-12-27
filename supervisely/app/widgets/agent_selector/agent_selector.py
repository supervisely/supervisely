from __future__ import annotations

from typing import Dict, List, Optional

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, ConditionalWidget
from supervisely.api.api import Api
from supervisely.api.agent_api import AgentInfo

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AgentSelector(ConditionalWidget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        def __init__(
            self,
            id,
            name,
            status: Literal["waiting", "running"] = None,
            disabled: bool = False,
        ) -> AgentSelector.Item:
            self._id = id
            self._name = name
            self._status = status
            self._disabled = disabled

        @property
        def id(self):
            return self._id

        @property
        def name(self):
            return self._name

        @property
        def status(self):
            return self._status

        @property
        def disabled(self):
            return self._disabled

        def to_json(self):
            return {
                "id": self._id,
                "name": self._name,
                "status": self._status,
                "disabled": self._disabled,
            }

    def __init__(
        self,
        team_id: int,
        only_running: bool = False,
        only_users_own: bool = False,
        has_gpu: bool = False,
        disable_unavaliable: bool = True,
        size: Literal["mini", "small", "large"] = None,
        multiple: bool = False,
        widget_id: str = None,
    ) -> AgentSelector:
        self._api: Api = Api.from_env()
        self._team_id = team_id
        self._changes_handled = False
        self._size = size
        self._multiple = multiple

        team_agents: List[AgentInfo] = self._api.agent.get_list(team_id)
        filtered_team_agents = self._filter_agents(
            team_agents,
            only_running,
            only_users_own,
            has_gpu,
            disable_unavaliable,
        )
        self._items = filtered_team_agents

        super().__init__(items=self._items, widget_id=widget_id, file_path=__file__)

    def _get_first_value(self) -> AgentSelector.Item:
        if self._items is not None and len(self._items) > 0:
            return self._items[0]
        return None

    def get_json_data(self) -> Dict:
        res = {
            "multiple": self._multiple,
            "items": None,
        }
        if self._items is not None:
            res["items"] = [item.to_json() for item in self._items]
        if self._size is not None:
            res["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        first_item = self._get_first_value()
        value = None
        if first_item is not None:
            value = first_item._id
        return {"value": value}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def get_selected_item(self) -> AgentSelector.Item:
        value = self.get_value()
        if value is None:
            return None
        for item in self.get_items():
            if item.id == value:
                return item
        return None

    def get_label(self):
        for item in self.get_items():
            if item._id == self.get_value():
                return item._name

    def get_labels(self):
        labels = []
        current_values = self.get_value()
        if not isinstance(current_values, list):
            current_values = [current_values]
        for item in self.get_items():
            if item.value in current_values:
                labels.append(item.label)
        return labels

    def get_items(self) -> List[AgentSelector.Item]:
        res = []
        if self._items is not None:
            res.extend(self._items)
        return res

    def set_value(self, value):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def update_items(
        self,
        show_running: bool = False,
        show_user: bool = False,
        has_gpu: bool = False,
        disable_unavaliable: bool = True,
    ):
        team_agents: List[AgentInfo] = self._api.agent.get_list(self._team_id)
        filtered_team_agents = self._filter_agents(
            team_agents, show_running, show_user, has_gpu, disable_unavaliable
        )
        self._items = filtered_team_agents
        DataJson()[self.widget_id]["items"] = [item.to_json() for item in self._items]
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(AgentSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self.get_selected_item()
            func(res)

        return _click

    def _filter_agents(
        self,
        agents: List[AgentInfo],
        only_running: bool = False,
        only_users_own: bool = False,
        has_gpu: bool = False,
        disable_unavaliable: bool = True,
    ) -> List[AgentInfo]:
        team_agents = agents
        if only_running:
            team_agents = [agent for agent in agents if agent.status == "running"]
        if only_users_own:
            user_info = self._api.user.get_my_info()
            team_agents = [agent for agent in agents if agent.user_id == user_info.id]
        if has_gpu:
            team_agents = [agent for agent in agents if agent.capabilities["app_gpu"]["enabled"]]

        filtered_team_agents = []
        for agent in team_agents:
            agent_item = AgentSelector.Item(
                id=agent.id,
                name=agent.name,
                status=agent.status,
                disabled=agent.status != "running" if disable_unavaliable else False,
            )
            filtered_team_agents.append(agent_item)
        return filtered_team_agents
