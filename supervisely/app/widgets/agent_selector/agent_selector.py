from typing import Dict, List

from supervisely import is_community
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class AgentSelector(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        team_id: int,
        show_only_gpu: bool = False,
        show_only_running: bool = True,
        compact: bool = False,
        widget_id=None,
    ):
        self._team_id = team_id
        self._show_any_status = not show_only_running
        self._show_public = True
        self._show_only_gpu = show_only_gpu
        self._check_network_status = True
        self._compact = compact

        self._is_community = is_community()

        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "teamId": self._team_id,
            "isCommunity": self._is_community,
        }

    def get_json_state(self) -> Dict:
        return {
            "agentId": None,
            "options": {
                "small": self._compact,
                "anyStatus": self._show_any_status,
                "showPublic": self._show_public,
                "needGpu": self._show_only_gpu,
                "checkAgentNetwork": self._check_network_status,
            },
        }

    def get_value(self) -> int:
        return StateJson()[self.widget_id]["agentId"]
    
    def set_value(self, agent_id: int) -> None:
        if not isinstance(agent_id, int):
            raise TypeError("Agent ID must be an integer.")
        StateJson()[self.widget_id]["agentId"] = agent_id
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(AgentSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self.get_value()
            func(res)

        return _click
