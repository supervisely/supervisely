import math
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class InputNumber(Widget):
    class Routes:
        CLICK = "button_clicked_cb"

    def __init__(
        self,
        value: int = 1,
        min: int = 1,
        max: int = 100,
        step: int = 1,
        size: str = "small",
        controls: bool = True,
        debounce: int = 300,
        show_loading: bool = True
    ):
        self._value = value
        self._min = min
        self._max = max
        self._step = step
        self._size = size
        self._controls = controls
        self._debounce = debounce

        self._loading = False
        self._disabled = False
        self._show_loading = show_loading
        self._hide = False

        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {
            "value": self._value,
            "min": self._min,
            "max": self._max,
            "step": self._step,
            "size": self._size,
            "controls": self._controls,
            "debounce": self._debounce
        }

    def get_json_state(self):
        return None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        DataJson()[self.widget_id]["value"] = self._value

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value
        DataJson()[self.widget_id]["min"] = self._min

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value
        DataJson()[self.widget_id]["max"] = self._max

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def show_loading(self):
        return self._show_loading

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled

    def hide(self):
        self._hide = True
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()

    def show(self):
        self._hide = False
        DataJson()[self.widget_id]["hide"] = self._hide
        DataJson().send_changes()
