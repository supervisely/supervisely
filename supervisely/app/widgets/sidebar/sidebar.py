from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class Sidebar(Widget):
    def __init__(
        self,
        left_content: Widget,
        right_content: Widget,
        width_percent: int = 25,
        widget_id: str = None,
        height: str = "100vh",
        standalone: bool = True,
        clear_main_panel_paddings: bool = False,
        show_close: bool = True,
        show_open: bool = True,
        sidebar_left_padding: str = None,
    ):
        self._left_content = left_content
        self._right_content = right_content
        self._width_percent = width_percent
        self._options = {
            "sidebarWidth": self._width_percent,
            "height": height,
            "clearMainPanelPaddings": clear_main_panel_paddings,
            "showOpen": show_open,
            "showClose": show_close,
            "sidebarLeftPadding": sidebar_left_padding,
        }
        super().__init__(widget_id=widget_id, file_path=__file__)

        if standalone:
            StateJson()["app_body_padding"] = "0px"
        StateJson()["menuIndex"] = "1"

    def get_json_data(self):
        return {"options": self._options}

    def get_json_state(self):
        return {
            "sidebarCollapsed": False,
        }

    def collapse(self):
        StateJson()[self.widget_id]["sidebarCollapsed"] = True
        StateJson().send_changes()

    def expand(self):
        StateJson()[self.widget_id]["sidebarCollapsed"] = False
        StateJson().send_changes()
