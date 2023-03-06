from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget
from supervisely.api.api import Api
from supervisely.app.widgets.select_sly_utils import _get_int_or_env


class TeamFilesSelector(Widget):
    def __init__(
        self,
        team_id: int,
        multiple_selection: bool = True,
        selection_file_type: Literal["folder", "file"] = None,
        hide_empty_table: bool = False,
        widget_id: str = None,
    ):
        self._api = Api()
        self._team_id = team_id
        self._multiple_selection = multiple_selection
        self._selection_file_type = selection_file_type
        self._hide_empty_table = hide_empty_table
        self._selected = [{"path": None, "type": None}]

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {"teamId": self._team_id, "multipleSelection": self._multiple_selection}

    def get_json_state(self) -> Dict:
        return {"selected": []}

    def get_selected_id(self):
        return DataJson()[self.widget_id]["teamId"]

    def get_selected_items(self):
        return StateJson()[self.widget_id]["selected"]
