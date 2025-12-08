from typing import Callable, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import SelectTeam, Widget, generate_id
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectWorkspace(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        default_id: int = None,
        team_id: int = None,
        compact: bool = False,
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._team_id = team_id
        self._compact = compact
        self._show_label = show_label
        self._size = size
        self._team_selector = None
        self._disabled = False
        self._changes_handled = False

        self._default_id = _get_int_or_env(self._default_id, "context.workspaceId")
        if self._default_id is not None:
            info = self._api.workspace.get_info_by_id(self._default_id, raise_error=True)
            self._team_id = info.team_id
        self._team_id = _get_int_or_env(self._team_id, "context.teamId")

        if compact is True:
            # If team_id is not provided in compact mode, start disabled
            if self._team_id is None:
                self._disabled = True
        else:
            # if self._show_label is False:
            #     logger.warn(
            #         "show_label can not be false if compact is True and default_id / team_id are not defined"
            #     )
            self._show_label = True
            self._team_selector = SelectTeam(
                default_id=self._team_id,
                show_label=True,
                size=self._size,
                widget_id=generate_id(),
            )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {}
        res["disabled"] = self._disabled
        res["teamId"] = self._team_id
        res["options"] = {
            "showLabel": self._show_label,
            "compact": self._compact,
            "filterable": True,
            "showWorkspace": True,
            "showTeam": False,
            "onlyAvailable": True,
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_team_id(self):
        if self._compact is True:
            return self._team_id
        else:
            return self._team_selector.get_selected_id()

    def set_team_id(self, team_id: int):
        """Set the team ID and update the UI. Automatically enables the widget if it was disabled."""
        self._team_id = team_id
        if self._compact is False and self._team_selector is not None:
            self._team_selector.set_team_id(team_id)
        else:
            DataJson()[self.widget_id]["teamId"] = team_id
            DataJson().send_changes()

        # Auto-enable the widget when team_id is set
        if self._disabled and team_id is not None:
            self.enable()

    def set_workspace_id(self, workspace_id: int):
        """Set the workspace ID and update the UI."""
        StateJson()[self.widget_id]["workspaceId"] = workspace_id
        StateJson().send_changes()

    def set_ids(self, team_id: int, workspace_id: int):
        """Set both team ID and workspace ID and update the UI."""
        self.set_team_id(team_id)
        self.set_workspace_id(workspace_id)

    def get_json_state(self) -> Dict:
        return {
            "workspaceId": self._default_id,
        }

    def get_selected_id(self):
        return StateJson()[self.widget_id]["workspaceId"]

    def disable(self):
        if self._compact is False:
            self._team_selector.disable()
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        if self._compact is False:
            self._team_selector.enable()
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def value_changed(self, func: Callable[[int], None]):
        """
        Decorator to handle workspace selection change event.
        The decorated function receives the selected workspace ID.

        :param func: Function to be called when workspace selection changes
        :type func: Callable[[int], None]
        """
        route_path = self.get_route_path(SelectWorkspace.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            workspace_id = self.get_selected_id()
            if workspace_id is not None:
                func(workspace_id)

        return _value_changed
