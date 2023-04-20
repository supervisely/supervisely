from supervisely.app import DataJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]


class Tag(Widget):
    # class Routes:
    #     CLOSE = "tag_close_cb"

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
    #     # add :closable="true"

    #     route_path = self.get_route_path(Tag.Routes.CLOSE)
    #     server = self._sly_app.get_server()
    #     self._click_handled = True

    #     @server.post(route_path)
    #     def _click():
    #         # maybe work with headers and store some values there r: Request
    #         if self.show_loading:
    #             self.loading = True
    #         try:
    #             func()
    #         except Exception as e:
    #             if self.show_loading and self.loading:
    #                 self.loading = False
    #             raise e
    #         if self.show_loading:
    #             self.loading = False

    #     return _click
