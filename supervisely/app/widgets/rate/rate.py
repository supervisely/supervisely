from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List


class Rate(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        max: int = 5,
        colors: List = ["#F7BA2A", "#F7BA2A", "#F7BA2A"],
        disabled: bool = False,
        allow_half: bool = False,
        texts: List[str] = [],
        show_text: bool = False,
        text_color: str = "#1F2D3D",
        void_color: str = "#C6D1DE",
        disabled_void_color: str = "#EFF2F7",
        widget_id: str = None,
    ):
        self._max = max
        self._colors = colors
        self._disabled = disabled
        self._allow_half = allow_half
        self._texts = texts
        self._show_text = show_text
        self._text_color = text_color
        self._void_color = void_color
        self._disabled_void_color = disabled_void_color
        self._changes_handled = False
        self._value = None
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "max": self._max,
            "colors": self._colors,
            "disabled": self._disabled,
            "allow_half": self._allow_half,
            "texts": self._texts,
            "show_text": self._show_text,
            "text_color": self._text_color,
            "void_color": self._void_color,
            "disabled_void_color": self._disabled_void_color,
        }

    def get_json_state(self):
        return {"value": self._value}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_current_value(self, value: int):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_max_value(self):
        return DataJson()[self.widget_id]["max"]

    def set_max_value(self, value: int):
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max
        DataJson().send_changes()

    def get_colors(self):
        return DataJson()[self.widget_id]["colors"]

    def set_colors(self, value: List[str]):
        self._colors = value
        DataJson()[self.widget_id]["colors"] = self._colors
        DataJson().send_changes()

    def disable(self):
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def get_disabled(self):
        return DataJson()[self.widget_id]["disabled"]

    def enable_allow_half(self):
        self._allow_half = True
        DataJson()[self.widget_id]["allow_half"] = self._allow_half
        DataJson().send_changes()

    def disable_allow_half(self):
        self._allow_half = False
        DataJson()[self.widget_id]["allow_half"] = self._allow_half
        DataJson().send_changes()

    def get_allow_half(self):
        return DataJson()[self.widget_id]["allow_half"]

    def get_texts(self):
        return DataJson()[self.widget_id]["texts"]

    def set_texts(self, value: List[str]):
        self._texts = value
        DataJson()[self.widget_id]["texts"] = self._texts
        DataJson().send_changes()

    def add_texts(self, value: List[str]):
        self._texts.extend(value)
        DataJson()[self.widget_id]["texts"] = self._texts
        DataJson().send_changes()

    def unable_show_text(self):
        self._show_text = True
        DataJson()[self.widget_id]["show_text"] = self._show_text
        DataJson().send_changes()

    def disable_show_text(self):
        self._show_text = False
        DataJson()[self.widget_id]["show_text"] = self._show_text
        DataJson().send_changes()

    def get_show_text(self):
        return DataJson()[self.widget_id]["show_text"]

    def get_text_color(self):
        return DataJson()[self.widget_id]["text_color"]

    def set_text_color(self, value: str):
        self._text_color = value
        DataJson()[self.widget_id]["text_color"] = self._text_color
        DataJson().send_changes()

    def get_void_color(self):
        return DataJson()[self.widget_id]["void_color"]

    def set_void_color(self, value: str):
        self._void_color = value
        DataJson()[self.widget_id]["void_color"] = self._void_color
        DataJson().send_changes()

    def get_disabled_void_color(self):
        return DataJson()[self.widget_id]["disabled_void_color"]

    def set_disabled_void_color(self, value: str):
        self._disabled_void_color = value
        DataJson()[self.widget_id]["disabled_void_color"] = self._disabled_void_color
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(Rate.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
