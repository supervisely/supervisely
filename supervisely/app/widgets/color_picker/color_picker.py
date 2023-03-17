from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Optional, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class ColorPicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        show_alpha: bool = False,
        color_format: Literal["hsl", "hsv", "hex", "rgb"] = "rgb",
        widget_id: str = None,
    ):
        self._show_alpha = show_alpha
        self._color_format = color_format
        self._color = None
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "show_alpha": self._show_alpha,
            "color_format": self._color_format,
        }

    def get_json_state(self):
        return {"color": self._color}

    def get_value(self):
        return StateJson()[self.widget_id]["color"]

    def value_changed(self, func):
        route_path = self.get_route_path(ColorPicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._color = res
            func(res)

        return _click
