try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import List
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class SelectAppSession(Widget):
    def __init__(
        self,
        team_id: int,
        tags: List[str],
        show_label: bool = False,
        size: Literal["large", "small", "mini"] = "mini",
        widget_id: str = None,
        operation: str = "or",
    ):
        self._session_id = None
        self._team_id = team_id
        self._tags = tags
        self._show_label = show_label
        self._size = size
        self._operation = operation

        if len(tags) < 1:
            raise ValueError("Parameter tags must be a list of strings, but got empty list")

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        data = {}
        data["teamId"] = self._team_id
        data["ssOptions"] = {
            "sessionTags": self._tags,
            "showLabel": self._show_label,
            "size": self._size,
            "sessionTagsOperation": self._operation,
        }
        return data

    def get_json_state(self):
        state = {}
        state["sessionId"] = self._session_id
        return state

    def get_selected_id(self):
        return StateJson()[self.widget_id]["sessionId"]
