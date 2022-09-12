import os
from typing import Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app.widgets import Widget, Select


class SelectTeam(Widget):
    def __init__(
        self,
        default_id: int = None,  # try automatically from env if None
        show_label: bool = True,
        size: Literal["large", "small", "mini"] = None,
        widget_id: str = None,
    ):
        self._default_id = default_id
        if self._default_id is None:
            self._default_id = os.environ.get("context.teamId")
        if self._default_id is not None:
            self._default_id = int(self._default_id)

        self._size = size
        self._show_label = show_label
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {
            "options": {
                "showLabel": self._show_label,
                "showWorkspace": False,
                "filterable": True,
                "onlyAvailable": True,
            }
        }
        if self._size is not None:
            res["options"]["size"] = self._size
        return res

    def get_json_state(self) -> Dict:
        return {
            "teamId": self._default_id,
        }
