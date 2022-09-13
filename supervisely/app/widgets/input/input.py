from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Input(Widget):
    def __init__(
            self,
            value: str = "",
            minlength: int = 0,
            maxlength: int = 1000,
            placeholder: str = "",
            size: Literal["mini", "small", "large"] = None,
            readonly: bool = False,
            widget_id: str = None
    ):
        self._value = value
        self._minlength = minlength
        self._maxlength = maxlength
        self._placeholder = placeholder
        self._size = size
        self._readonly = readonly
        self._widget_id = widget_id

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "minlength": self._minlength,
            "maxlength": self._maxlength,
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def is_readonly(self):
        return DataJson()[self.widget_id]["readonly"]

    def enable_readonly(self):
        DataJson()[self.widget_id]["readonly"] = True
        DataJson().send_changes()

    def disable_readonly(self):
        DataJson()[self.widget_id]["readonly"] = False
        DataJson().send_changes()
