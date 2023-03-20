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
        start: str = "09:00",
        step: str = "00:15",
        end: str = "22:30",
        placeholder: str = None,
        size: Literal["large", "small", "mini"] = None,
        readonly: bool = False,
        disabled: bool = False,
        editable: bool = True,
        clearable: bool = True,
        widget_id: str = None,
    ):
        self._start = start
        self._step = step
        self._end = end
        self._placeholder = placeholder
        self._size = size
        self._readonly = readonly
        self._disabled = disabled
        self._editable = editable
        self._clearable = clearable

        self._changes_handled = False
        self._value = None
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "readonly": self._readonly,
            "disabled": self._disabled,
            "editable": self._editable,
            "clearable": self._clearable,
            "picker_options": {"start": self._start, "step": self._step, "end": self._end},
        }

    def get_json_state(self):
        return {"value": self._value}

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
