from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Sidebar(Widget):
    def __init__(
        self,
        left_pane: Widget,
        right_pane: Widget,
        width_percent: int = 25,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._left_pane = left_pane
        self._right_pane = right_pane
        self._width_percent = width_percent
        self._options = {"sidebarWidth": self._width_percent}
        StateJson()["app_body_padding"] = "0px"
        StateJson()["menuIndex"] = "1"

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}
