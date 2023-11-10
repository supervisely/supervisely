from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, Text, Empty
from typing import Optional, List

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
        color_format: Literal["hex", "hsl", "hsv", "rgb"] = "hex",
        compact: bool = False,
        widget_id: str = None,
    ):
        self._show_alpha = show_alpha
        self._color_format = color_format
        self._changes_handled = False
        self._color_info = Empty()
        self._compact = compact

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
            "compact": self._compact,
        }

    def get_json_state(self):
        return {"color": self._color}

    def get_value(self):
        return StateJson()[self.widget_id]["color"]

    def set_value(self, value: Optional[str or List[int, int, int]]):
        self._color = value
        if isinstance(self._color, list) and len(self._color) == 3 and self._color_format == "rgb":
            if (
                isinstance(self._color[0], int)
                and isinstance(self._color[1], int)
                and isinstance(self._color[2], int)
            ):
                self._color = f"rgb({self._color[0]}, {self._color[1]}, {self._color[2]})"
        if (
            (self._color_format == "hex" and self._color[0] != "#")
            or (self._color_format == "hsl" and self._color[0:3] != "hsl")
            or (self._color_format == "hsv" and self._color[0:3] != "hsv")
            or (self._color_format == "rgb" and self._color[0:3] != "rgb")
        ):
            raise ValueError(
                f"Incorrect input value format: {self._color}, {self._color_format} format should be, check your input data"
            )
        StateJson()[self.widget_id]["color"] = self._color
        StateJson().send_changes()

    def is_show_alpha_enabled(self):
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
