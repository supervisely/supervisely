from typing import List
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Container(Widget):
    def __init__(
        self,
        widgets: List[Widget] = [],
        gap: int = 10,
        widget_id: str = None,
    ):
        self._widgets = widgets
        self._gap = gap
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
