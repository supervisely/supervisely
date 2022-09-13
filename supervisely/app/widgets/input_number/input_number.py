from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class InputNumber(Widget):
    def __init__(
        self,
        value: int = 1,
        min: int = 1,
        max: int = 100,
        step: int = 1,
        size: str = "small",
        controls: bool = True,
        debounce: int = 300,
        widget_id: str = None
    ):
        self._value = value
        self._min = min
        self._max = max
        self._step = step
        self._size = size
        self._controls = controls
        self._debounce = debounce

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "min": self._min,
            "max": self._max,
            "step": self._step,
            "size": self._size,
            "controls": self._controls,
            "debounce": self._debounce
        }

    def get_json_state(self):
        return {"value": self._value}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()


    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value
        DataJson()[self.widget_id]["min"] = self._min
        DataJson().send_changes()


    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max
        DataJson().send_changes()

