import os
from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app.widgets import Widget
from supervisely.api.api import Api


class SelectWorkspace(Widget):
    def __init__(
        self,
        default_id: int = None,  # try automatically from env if None
        team_id: int = None,
        size: Literal["large", "small", "mini"] = None,
        show_label: bool = True,
        widget_id: str = None,
    ):
        self._default_id = os.environ.get("context.workspaceId", default_id)
        if self._default_id is not None:
            self._default_id = int(self._default_id)

        self._team_id = team_id
        if self._team_id is None:
            self._team_id = int(os.environ.get("context.teamId"))
        if self._team_id is None:
            if self._default_id is not None:
                ws_info = Api().workspace.get_info_by_id(self._default_id)
                self._team_id = ws_info.team_id
            else:
                raise ValueError(
                    "team_id has to be defined as argument or as in env variable 'context.teamId'"
                )

        self._size = size
        self._show_label = show_label
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {
            "options": {
                "showLabel": self._show_label,
                "showTeam": False,
                "showWorkspace": True,
                "filterable": True,
                "onlyAvailable": True,
            }
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "teamId": self._team_id,
            "workspaceId": self._default_id,
        }
