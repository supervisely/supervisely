from typing import Dict, List

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class AgentSelector(Widget):
    def __init__(
        self,
        team_id: int,
        any_status: bool = True,
        show_public: bool = False,
        has_gpu: bool = False,
        only_running: bool = False,
        compact: bool = False,
        widget_id=None,
    ):
        self._team_id = team_id
        self._any_status = any_status
        self._show_public = show_public
        self._has_gpu = has_gpu
        self._only_running = only_running
        self._compact = compact

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"teamId": self._team_id}

    def get_json_state(self) -> Dict:
        return {
            "agentId": None,
            "options": {
                # "small": self._compact,
                # "anyStatus": self._any_status,
                # "showPublic": self._show_public,
                # "needGpu": self._has_gpu,
                # "checkAgentNetwork": self._only_running,
            },
        }

    def get_value(self):
        return StateJson()[self.widget_id]["agentId"]

    # def value_changed(self, func):
    #     route_path = self.get_route_path(Select.Routes.VALUE_CHANGED)
    #     server = self._sly_app.get_server()
    #     self._changes_handled = True

    #     @server.post(route_path)
    #     async def _click():
    #         res = self.get_value()
    #         func(res)

    #     return _click
