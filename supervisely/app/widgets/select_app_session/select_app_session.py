try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import List
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class SelectAppSession(Widget):
    """Widget to select an app session by tags within a team (returns selected session ID)."""

    class Routes:
        """Callback route names used by the widget frontend to notify Python."""

        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        team_id: int,
        tags: List[str],
        show_label: bool = False,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
        operation: str = "or",
    ):
        """:param team_id: Team ID to list sessions from.
        :type team_id: int
        :param tags: List of tag names to filter sessions.
        :type tags: List[str]
        :param show_label: If True, show label.
        :type show_label: bool
        :param size: Size: "large", "small", or "mini".
        :type size: Literal["large", "small", "mini"], optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param operation: Tag filter: "or" or "and".
        :type operation: str

        :raises ValueError: If tags is not a non-empty list.
        """
        self._session_id = None
        self._team_id = team_id
        self._tags = tags
        self._show_label = show_label
        self._size = size
        self._operation = operation

        if not isinstance(tags, list):
            raise ValueError("Parameter tags must be a list of strings")
        
        if len(tags) < 1:
            raise ValueError("Parameter tags must be a list of strings, but got empty list")

        super().__init__(widget_id=widget_id, file_path=__file__)

    def value_changed(self, func):
        route_path = self.get_route_path(SelectAppSession.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_selected_id()
            func(res)

        return _click

    def get_json_data(self):
        data = {}
        data["teamId"] = self._team_id
        data["ssOptions"] = {
            "sessionTags": self._tags,
            "showLabel": self._show_label,
            "sessionTagsOperation": self._operation,
        }
        if self._size is not None:
            data["ssOptions"]["size"] = self._size
        return data

    def get_json_state(self):
        state = {}
        state["sessionId"] = self._session_id
        return state

    def set_session_id(self, session_id: int):
        self._session_id = session_id
        StateJson()[self.widget_id]["sessionId"] = self._session_id
        StateJson().send_changes()

    def get_selected_id(self):
        return StateJson()[self.widget_id]["sessionId"]
