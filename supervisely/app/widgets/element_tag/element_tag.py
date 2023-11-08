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
        # tags: list = [],
        text: str = "",
        type: Literal["primary", "gray", "success", "warning", "danger", None] = None,
        hit: bool = False,
        color: str = "",
        closable: bool = False,
        close_transition: bool = False,
        hide: bool = False,
        widget_id: str = None,
    ):
        self._text = text
        self._validate_type(type)
        self._type = type
        self._color = color
        self._closable = closable
        self._close_transition = close_transition
        self._hit = hit
        self._is_hide = hide
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
            "closable": self._closable,
            "close_transition": self._close_transition,
        }

    # def get_json_state(self):
    #     return {"clicked_tag": self._clicked_tag}

    def get_json_state(self):
        return {}

    # def get_clicked_tag(self):
    #     return StateJson()[self.widget_id]["clicked_tag"]

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

    def get_color(self):
        return DataJson()[self.widget_id]["color"]

    def set_color(self, value: str):
        DataJson()[self.widget_id]["color"] = value
        DataJson().send_changes()

    # def close_tag(self, func):
    #     route_path = self.get_route_path(Tag.Routes.CLOSE)
    #     server = self._sly_app.get_server()
    #     self._tag_close = True

    #     @server.post(route_path)
    #     def _click():
    #         res = self.get_clicked_tag()
    #         func(res)

    #     return _click
