from typing import Optional

from supervisely.api.api import Api
from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class NewExperiment(Widget):

    class Routes:
        VISIBLE_CHANGED = "visible_changed_cb"

    def __init__(
        self,
        workspace_id: int,
        team_id: Optional[int] = None,
        redirect_to_session: bool = False,
        widget_id: Optional[str] = None,
    ):
        self._api = Api()
        self._workspace_id = workspace_id
        self._team_id = team_id
        self._changes_handled = False
        if self._team_id is None:
            workspace = self._api.workspace.get_info_by_id(self._workspace_id)
            if workspace is None:
                raise ValueError(f"Workspace with ID {self._workspace_id} not found.")
            self._team_id = workspace.team_id
        self._redirect_to_session = redirect_to_session
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "options": {
                "redirectToSession": self._redirect_to_session,
            },
            "workspaceId": self._workspace_id,
            "teamId": self._team_id,
        }

    def get_json_state(self):
        return {"visible": False}

    @property
    def visible(self) -> bool:
        return StateJson()[self.widget_id]["visible"]

    @visible.setter
    def visible(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Visible must be a boolean value.")
        StateJson()[self.widget_id]["visible"] = value
        StateJson().send_changes()

    def visible_changed(self, func):
        route_path = self.get_route_path(NewExperiment.Routes.VISIBLE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            func(self.visible)

        return _click
