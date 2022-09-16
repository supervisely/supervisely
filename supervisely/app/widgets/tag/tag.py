from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Tag(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: str = "",
        type: Literal["primary", "gray", "success", "warning", "danger"] = None,
        hit: bool = False,
        widget_id: str = None
    ):
        self._value = value
        self._type = type
        self._hit = hit
        self._widget_id = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "type": self._type,
            "hit": self._hit,
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value: str):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_type(self, value: str):
        if value not in ["primary", "gray", "success", "warning", "danger"]:
            return
        DataJson()[self.widget_id]["value"] = value
        DataJson().send_changes()

    def get_type(self):
        return DataJson()[self.widget_id]["value"]

    def is_highlighted(self):
        return DataJson()[self.widget_id]["hit"]

    def enable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = True
        DataJson().send_changes()

    def disable_border_highlighting(self):
        DataJson()[self.widget_id]["hit"] = False
        DataJson().send_changes()
