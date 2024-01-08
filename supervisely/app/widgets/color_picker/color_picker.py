from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Empty, Widget


class ColorPicker(Widget):
    """ColorPicker is a color selector supporting multiple color formats.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/input/colorpicker>`_
        (including screenshots and examples).

    :param show_alpha: if True, alpha channel will be shown
    :type show_alpha: bool
    :param color_format: color format, one of: hex, hsl, hsv, rgb
    :type color_format: Literal["hex", "hsl", "hsv", "rgb"]
    :param compact: if True, compact mode will be enabled
    :type compact: bool
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import ColorPicker

        color_picker = ColorPicker(color_format="rgb")
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        show_alpha: Optional[bool] = False,
        color_format: Optional[Literal["hex", "hsl", "hsv", "rgb"]] = "hex",
        compact: Optional[bool] = False,
        widget_id: Optional[str] = None,
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

    def get_json_data(self) -> Dict[str, Union[bool, str]]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - show_alpha: if True, alpha channel will be shown
            - color_format: color format, one of: hex, hsl, hsv, rgb
            - compact: if True, compact mode will be enabled

        :return: dictionary with widget data
        :rtype: Dict[str, Union[bool, str]]
        """
        return {
            "show_alpha": self._show_alpha,
            "color_format": self._color_format,
            "compact": self._compact,
        }

    def get_json_state(self) -> Dict[str, Union[str, List[int]]]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - color: current color

        :return: dictionary with widget state
        :rtype: Dict[str, Union[str, List[int]]]
        """
        return {"color": self._color}

    def get_value(self) -> Union[str, List[int]]:
        """Returns current color.

        :return: current color
        :rtype: Union[str, List[int]]
        """
        return StateJson()[self.widget_id]["color"]

    def set_value(self, value: Optional[Union[str, List[int]]]) -> None:
        """Sets current color.

        :param value: current color
        :type value: Union[str, List[int]]]
        """
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

    def is_show_alpha_enabled(self) -> bool:
        """Returns True if alpha channel is shown, False otherwise.

        :return: True if alpha channel is shown, False otherwise
        :rtype: bool
        """
        return DataJson()[self.widget_id]["show_alpha"]

    def disable_show_alpha(self) -> None:
        """Disables alpha channel."""
        self._show_alpha = False
        DataJson()[self.widget_id]["show_alpha"] = self._show_alpha
        DataJson().send_changes()

    def enable_show_alpha(self) -> None:
        """Enables alpha channel."""
        self._show_alpha = True
        DataJson()[self.widget_id]["show_alpha"] = self._show_alpha
        DataJson().send_changes()

    def get_color_format(self) -> Literal["hex", "hsl", "hsv", "rgb"]:
        """Returns current color format.

        :return: current color format
        :rtype: Literal["hex", "hsl", "hsv", "rgb"]
        """
        return DataJson()[self.widget_id]["color_format"]

    def set_color_format(self, value: Literal["hex", "hsl", "hsv", "rgb"]) -> None:
        """Sets current color format.

        :param value: current color format
        :type value: Literal["hex", "hsl", "hsv", "rgb"]
        """
        self._color_format = value
        DataJson()[self.widget_id]["color_format"] = self._color_format
        DataJson().send_changes()

    def value_changed(self, func: Callable[[Union[str, List[int]]], Any]) -> Callable[[], None]:
        """Decortator for function that will be called when color is changed.

        :param func: function that will be called when color is changed
        :type func: Callable[[Union[str, List[int]]], Any]
        :return: decorated function
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(ColorPicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._color = res
            func(res)

        return _click
