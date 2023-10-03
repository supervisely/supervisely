import re
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class ColorPicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        color: Union[str, tuple, list, None] = None,  # hex or rgb
        color_format: Literal["hex", "rgb"] = "rgb",
        widget_id: str = None,
    ):
        self._color_format = color_format
        self._color = self._prepare_color(color, self._color_format) if color else None
        self._changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "colorFormat": self._color_format,
            "showAlpha": False,
        }

    def get_json_state(self):
        return {"color": self._color}

    @property
    def hex_color(self) -> Union[str, None]:
        return self._get_value(color_format="hex")

    @property
    def rgb_color(self) -> Union[str, None]:
        color = self._get_value(color_format="rgb")
        return self._get_rgb_sequence(color)

    def get_value(self) -> Union[str, None]:
        self._color = self._get_value(self._color_format)
        if self._color_format == "rgb":
            return self._get_rgb_sequence(self._color)
        return self._color

    def set(self, color: str) -> None:
        if not self._is_hex(color) and not self._is_rgb(color):
            raise ValueError("Invalid color format")
        self._color = self._prepare_color(color, self._color_format)
        StateJson()[self.widget_id]["color"] = self._color
        StateJson().send_changes()

    def clear_value(self) -> None:
        self._color = None
        StateJson()[self.widget_id]["color"] = self._color
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(ColorPicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_value()
            func(res)

        return _value_changed

    def _get_value(self, color_format: Literal["hex", "rgb"]) -> Union[str, None]:
        color = StateJson()[self.widget_id]["color"]
        color = self._prepare_color(color, color_format)
        return color

    def _prepare_color(self, color, color_format) -> Union[str, None]:
        if color_format == "hex" and self._is_rgb(color):
            color = self._rgb_to_hex(color)
        elif color_format == "rgb":
            if self._is_hex(color):
                color = self._hex_to_rgb(color)
            elif self._is_rgb(color):
                color = self._get_rgb_str(self._get_rgb_sequence(color))
        return color

    def _hex_to_rgb(self, hex: str) -> str:
        if not self._is_hex(hex):
            raise ValueError("Invalid hex color format")
        hex = hex.lstrip("#")
        rgb = [int(hex[0:2], 16), int(hex[2:4], 16), int(hex[4:6], 16)]
        rgb = self._get_rgb_str(rgb)
        return rgb

    def _rgb_to_hex(self, rgb: Union[tuple, list, str]) -> str:
        rgb = self._get_rgb_sequence(rgb)
        if rgb is None:
            raise ValueError("Invalid rgb color format")
        r, g, b = rgb
        return "#%02x%02x%02x" % (r, g, b)

    def _is_hex(self, hex: str) -> bool:
        if type(hex) is not str:
            return False
        match = re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", hex)
        return match is not None

    def _is_rgb(self, rgb: Union[tuple, list, str]) -> bool:
        rgb = self._get_rgb_sequence(rgb)
        return rgb is not None

    def _get_rgb_sequence(self, rgb: Union[tuple, list, str]) -> Union[list, None]:
        if type(rgb) is str:
            if rgb.startswith("rgba") or rgb.startswith("RGBA"):
                raise ValueError("Alpha channel is not supported")
            if rgb.startswith("rgb") or rgb.startswith("RGB"):
                rgb = list(map(int, rgb[4:-1].split(",")))
            else:
                return None
        if type(rgb) not in [list, tuple]:
            return None
        if len(rgb) == 3 and all(type(val) is int for val in rgb):
            return rgb
        return None

    def _get_rgb_str(self, rgb: Union[tuple, list]) -> Union[str, None]:
        if rgb is None:
            return None
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
