from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class TimePicker(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: str = "asd",
        start: str = "09:00",
        step: str = "00:15",
        end: str = "22:30",
        placeholder: str = None,
        size: Literal["large", "small", "mini"] = None,
        readonly: bool = False,
        editable: bool = True,
        clearable: bool = True,
        widget_id: str = None,
    ):
        self._validate_value(value)
        self._value = value
        self._start = start
        self._step = step
        self._end = end
        self._placeholder = placeholder
        self._size = size
        self._readonly = readonly
        self._editable = editable
        self._clearable = clearable

        self._changes_handled = False
        # self._value = None
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "editable": self._editable,
            "clearable": self._clearable,
            "picker_options": {"start": self._start, "step": self._step, "end": self._end},
        }

    def get_json_state(self):
        return {"value": self._value}

    def _validate_value(self, value):
        if value is None or value == "":
            return
        if not isinstance(value, str):
            raise TypeError("value must be str type")

        val_split = value.split(":")
        if len(val_split) != 2:
            raise ValueError("value must be in format 'hh:mm'")
        hh, mm = val_split
        if len(hh) != 2 or len(mm) != 2:
            raise ValueError("value must be in format 'hh:mm'")
        if not hh[0].isdigit() or not hh[1].isdigit() or not mm[0].isdigit() or not mm[1].isdigit():
            raise ValueError("value must be in format 'hh:mm'")

    def set_value(self, value: str):
        self._validate_value(value)
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def get_picker_options(self):
        return DataJson()[self.widget_id]["picker_options"]

    def set_start(self, value: str):
        self._start = value
        DataJson()[self.widget_id]["picker_options"]["start"] = self._start
        DataJson().send_changes()

    def set_end(self, value: str):
        self._end = value
        DataJson()[self.widget_id]["picker_options"]["end"] = self._end
        DataJson().send_changes()

    def set_step(self, value: str):
        self._step = value
        DataJson()[self.widget_id]["picker_options"]["step"] = self._step
        DataJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(TimePicker.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
