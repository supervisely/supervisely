from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]


class ElementTag(Widget):
    class Routes:
        CLOSE = "tag_close_cb"

    def __init__(
        self,
        text: str = "",
        type: Literal["primary", "gray", "success", "warning", "danger", None] = None,
        hit: bool = False,
        color: str = "",
        widget_id: str = None,
    ):
        self._text = text
        self._validate_type(type)
        self._type = type
        self._color = color
        self._hit = hit
        self._tag_close = False
        self._clicked_tag = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_type(self, value):
        if value is None:
            return
        if value not in SUPPORTED_TAG_WIDGET_TYPES:
            raise ValueError(
                "type = {!r} is unknown, should be one of {}".format(
                    value, SUPPORTED_TAG_WIDGET_TYPES
                )
            )

    def get_json_data(self):
        return {
            "text": self._text,
            "type": self._type,
            "hit": self._hit,
            "color": self._color,
        }

    def get_json_state(self):
        return {}

    @property
    def text(self):
        return self._text

    @property
    def type(self):
        return self._type

    @property
    def hit(self):
        return self._hit

    @property
    def color(self):
        return self._color

    def set_text(self, value: str):
        DataJson()[self.widget_id]["text"] = value
        DataJson().send_changes()

    def get_text(self):
        return DataJson()[self.widget_id]["text"]

    def set_type(self, value: Literal["primary", "gray", "success", "warning", "danger"]):
        self._validate_type(value)
        DataJson()[self.widget_id]["type"] = value
        DataJson().send_changes()

    def get_type(self):
        return DataJson()[self.widget_id]["type"]

    def is_border_highlighted(self):
        return DataJson()[self.widget_id]["hit"]

    def enable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = True
        DataJson().send_changes()

    def disable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = False
        DataJson().send_changes()

    def set_color(self, value: str):
        DataJson()[self.widget_id]["color"] = value
        DataJson().send_changes()

    def get_color(self):
        return DataJson()[self.widget_id]["color"]
