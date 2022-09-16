from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

SUPPORTED_TAG_WIDGET_TYPES = ["primary", "gray", "success", "warning", "danger"]


class Tag(Widget):
    def __init__(
            self,
            text: str = "",
            type: Literal["primary", "gray", "success", "warning", "danger"] = None,
            hit: bool = False,
            widget_id: str = None
    ):
        self._text = text
        self._validate_type(type)
        self._type = type
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
            "hit": self._hit
        }

    def get_json_state(self):
        return {}

    def set_text(self, value: str):
        DataJson()[self.widget_id]["text"] = value
        DataJson().send_changes()

    def get_text(self):
        return StateJson()[self.widget_id]["text"]

    def set_type(self, value: Literal["primary", "gray", "success", "warning", "danger"]):
        self._validate_type(value)
        DataJson()[self.widget_id]["value"] = value
        DataJson().send_changes()

    def get_type(self):
        return DataJson()[self.widget_id]["value"]

    def is_border_highlighted(self):
        return DataJson()[self.widget_id]["hit"]

    def enable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = True
        DataJson().send_changes()

    def disable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = False
        DataJson().send_changes()
