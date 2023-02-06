try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import supervisely as sly
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api


class ModelInfo(Widget):
    def __init__(
        self,
        session_id: int = None,
        team_id: int = None,
        widget_id: str = None,
    ):
        self._api = Api()
        self._session_id = session_id
        self._team_id = team_id

        if self._team_id is None:
            self._team_id = sly.env.team_id()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        data = {}
        data["teamId"] = self._team_id
        if self._session_id is not None:
            data["model_info"] = self._api.task.send_request(
                self._session_id, "get_session_info", data={}
            )
        else:
            data["model_info"] = None
        if self._session_id is not None:
            data["model_connected"] = True
        else:
            data["model_connected"] = False
        return data

    def get_json_state(self):
        state = {}
        state["sessionId"] = self._session_id
        return state

    def set_session_id(self, session_id):
        self._session_id = session_id
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()
