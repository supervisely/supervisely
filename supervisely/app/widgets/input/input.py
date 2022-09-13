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
        self._type = "text"
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
        return StateJson()[self.widget_id]["currentFrame"]

    @property
    def minlength(self):
        return self._minlength

    @minlength.setter
    def minlength(self, value):
        self._minlength = value
        DataJson()[self.widget_id]["minlength"] = self._minlength
        DataJson().send_changes()

    @property
    def maxlength(self):
        return self._maxlength

    @maxlength.setter
    def maxlength(self, value):
        self._maxlength = value
        DataJson()[self.widget_id]["maxlength"] = self._maxlength
        DataJson().send_changes()

    @property
    def readonly(self):
        return self._readonly

    @readonly.setter
    def readonly(self, value):
        self._readonly = value
        DataJson()[self.widget_id]["readonly"] = self._readonly
        DataJson().send_changes()
