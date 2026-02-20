from typing import Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class InputNumber(Widget):
    """Numeric input widget with min/max/step/precision and value-changed callback."""

    class Routes:
        """Callback route names used by the widget frontend to notify Python."""

        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        value: Union[int, float] = 1,
        min: Union[int, float, None] = None,
        max: Union[int, float, None] = None,
        step: Union[int, float] = 1,
        size: str = "small",
        controls: bool = True,
        debounce: int = 300,
        precision: int = 0,
        widget_id: str = None,
        width: int = None,
    ):
        """
        :param value: Initial numeric value.
        :type value: Union[int, float]
        :param min: Minimum value. None for no limit.
        :type min: Union[int, float, None], optional
        :param max: Maximum value. None for no limit.
        :type max: Union[int, float, None], optional
        :param step: Step increment.
        :type step: Union[int, float]
        :param size: Size ("small", "large", etc.).
        :type size: str
        :param controls: If True, show +/- buttons.
        :type controls: bool
        :param debounce: Debounce delay in ms for value-changed callback.
        :type debounce: int
        :param precision: Decimal places for float values.
        :type precision: int
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param width: Width in pixels. None for auto.
        :type width: int, optional
        """
        self._value = value
        self._min = min
        self._max = max
        self._step = step
        self._size = size
        self._controls = controls
        self._debounce = debounce
        self._precision = precision
        self._width = width
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "min": self._min,
            "max": self._max,
            "step": self._step,
            "size": self._size,
            "controls": self._controls,
            "debounce": self._debounce,
            "precision": self._precision,
        }

    def get_json_state(self):
        return {"value": self._value}

    @property
    def value(self):
        self._value = StateJson()[self.widget_id]["value"]
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_value(self):
        self._value = StateJson()[self.widget_id]["value"]
        return self._value

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

    def value_changed(self, func):
        route_path = self.get_route_path(InputNumber.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_value()
            self._value = res
            func(res)

        return _click
