from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Optional, Dict

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
        placeholder: str = "Select time",
        size: Literal["large", "small", "mini"] = None,
        popper_class: str = None,
        widget_id: str = None,
    ):
        self._start = start
        self._step = step
        self._end = end
        self._placeholder = placeholder
        self._size = size
        self._popper_class = popper_class

        self._changes_handled = False
        self._value = None
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "placeholder": self._placeholder,
            "size": self._size,
            "popper_class": self._popper_class,
            "picker_options": {"start": self._start, "step": self._step, "end": self._end},
        }

    def get_json_state(self):
        return {"value": self._value}

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

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
