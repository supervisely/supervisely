from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

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
        color_format: Literal["hsl", "hsv", "hex", "rgb"] = "hex",
        widget_id: str = None,
    ):
        self._show_alpha = show_alpha
        self._color_format = color_format
        self._changes_handled = False

        if self._color_format not in ["hsl", "hsv", "hex", "rgb"]:
            raise TypeError(
                f"Incorrect color format: {self._color_format}, only hsl, hsv, hex, rgb are possible"
            )

        if self._color_format == "hex":
            self._color = "#20a0ff"
        elif self._color_format == "hsl":
            self._color = "hsl(205, 100%, 56%)"
        elif self._color_format == "hsv":
            self._color = "hsv(205, 87%, 100%)"
        else:
            self._color = "rgb(32, 160, 255)"

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

    def set_value(self, value: str):
        self._color = value
        StateJson()[self.widget_id]["color"] = self._color
        StateJson().send_changes()

    def check_show_alpha(self):
        return DataJson()[self.widget_id]["show_alpha"]

    def disable_show_alpha(self):
        self._show_alpha = False
        DataJson()[self.widget_id]["show_alpha"] = self._show_alpha
        DataJson().send_changes()

    def enable_show_alpha(self):
        self._show_alpha = True
        DataJson()[self.widget_id]["show_alpha"] = self._show_alpha
        DataJson().send_changes()

    def get_color_format(self):
        return DataJson()[self.widget_id]["color_format"]

    def set_color_format(self, value: str):
        self._color_format = value
        DataJson()[self.widget_id]["color_format"] = self._color_format
        DataJson().send_changes()

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
