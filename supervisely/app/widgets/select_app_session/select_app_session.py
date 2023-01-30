from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class SelectAppSession(Widget):
    def __init__(
        self,
        team_id: int = None,
        show_label: bool = False,
        size: Literal["large", "small", "mini"] = "mini",
        widget_id: str = None,
    ):
        self._api = Api()
        self._session_id = None
        self._team_id = team_id
        self._show_label = show_label
        self._size = size

        if self._team_id is None:
            self._team_id = sly.env.team_id()
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        data = {}
        data["ssOptions"] = {
            "sessionTags": ["deployed_nn", "deployed_nn_keypoints"],
            "showLabel": self._show_label,
            "size": self._size,
            "sessionTagsCombination": False,
        }
        data["teamId"] = self._team_id
        return data

    def get_json_state(self):
        state = {}
        state["sessionId"] = self._session_id
        return state

    def get_selected_id(self):
        return StateJson()[self.widget_id]["sessionId"]
