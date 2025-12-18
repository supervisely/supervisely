from typing import Callable, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.api.api import Api
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectTeam(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        default_id: int = None,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._show_label = show_label
        self._size = size
        self._changes_handled = False

        self._default_id = _get_int_or_env(self._default_id, "context.teamId")
        if self._default_id is not None:
            self._api.team.get_info_by_id(self._default_id, raise_error=True)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["options"] = {
            "showLabel": self._show_label,
            "filterable": True,
            "showWorkspace": False,
            "showTeam": True,
            "onlyAvailable": True,
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "teamId": self._default_id,
        }

    def get_selected_id(self):
        return StateJson()[self.widget_id].get("teamId")

    def set_team_id(self, team_id: int):
        """Set the selected team ID.

        :param team_id: Team ID to select
        :type team_id: int
        """
        StateJson()[self.widget_id]["teamId"] = team_id
        StateJson().send_changes()

    def value_changed(self, func: Callable[[int], None]):
        """
        Decorator to handle team selection change event.
        The decorated function receives the selected team ID.

        :param func: Function to be called when team selection changes
        :type func: Callable[[int], None]
        """
        route_path = self.get_route_path(SelectTeam.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            team_id = self.get_selected_id()
            if team_id is not None:
                func(team_id)

        return _value_changed
