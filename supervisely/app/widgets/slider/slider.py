from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Slider(Widget):
    def __init__(
            self,
            value: Union[int, List[int]] = 0,
            min: int = 0,
            max: int = 100,
            step: int = 1,
            show_input: bool = False,
            show_input_controls: bool = False,
            show_stops: bool = False,
            show_tooltip: bool = True,
            range: bool = False,
            vertical: bool = False,
            height: int = None,
            widget_id: str = None
    ):
        self._value = value
        self._min = min
        self._max = max
        self._step = step
        self._show_input = show_input
        self._show_input_controls = show_input_controls if show_input else False
        self._show_stops = show_stops
        self._show_tooltip = show_tooltip
        self._range = False if show_input else range
        self._vertical = vertical
        self._height = f"{height}px" if vertical else None
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
        "min": self._min,
        "max": self._max,
        "step": self._step,
        "showInput": self._show_input,
        "showInputControls": self._show_input_controls,
        "showStops": self._show_stops,
        "showTooltip": self._show_tooltip,
        "range": self._range,
        "vertical": self._vertical,
        "height": self._height
        }

    def get_json_state(self):
        return {"value": self._value}

    def set_value(self, value: int):
        StateJson()[self.widget_id]["value"] = value
        StateJson().send_changes()

    def get_value(self):
        return StateJson()[self.widget_id]["value"]

    def set_min(self, value: int):
        DataJson()[self.widget_id]["min"] = value
        DataJson().send_changes()

    def get_min(self):
        return DataJson()[self.widget_id]["min"]

    def set_max(self, value: int):
        DataJson()[self.widget_id]["max"] = value
        DataJson().send_changes()

    def get_max(self):
        return DataJson()[self.widget_id]["max"]

    def set_step(self, value: int):
        DataJson()[self.widget_id]["step"] = value
        DataJson().send_changes()

    def get_step(self):
        return DataJson()[self.widget_id]["step"]

    def is_input_enabled(self):
        return DataJson()[self.widget_id]["showInput"]

    def show_input(self):
        DataJson()[self.widget_id]["showInput"] = True
        DataJson().send_changes()

    def hide_input(self):
        DataJson()[self.widget_id]["showInput"] = False
        DataJson().send_changes()

    def is_input_controls_enabled(self):
        return DataJson()[self.widget_id]["showInputControls"]

    def show_input_controls(self):
        DataJson()[self.widget_id]["showInputControls"] = True
        DataJson().send_changes()

    def hide_input_controls(self):
        DataJson()[self.widget_id]["showInputControls"] = False
        DataJson().send_changes()

    def is_step_enabled(self):
        return DataJson()[self.widget_id]["showStops"]

    def show_steps(self):
        DataJson()[self.widget_id]["showStops"] = True
        DataJson().send_changes()

    def hide_steps(self):
        DataJson()[self.widget_id]["showStops"] = False
        DataJson().send_changes()

    def is_tooltip_enabled(self):
        return DataJson()[self.widget_id]["showTooltip"]

    def show_tooltip(self):
        DataJson()[self.widget_id]["showTooltip"] = True
        DataJson().send_changes()

    def hide_tooltip(self):
        DataJson()[self.widget_id]["showTooltip"] = False
        DataJson().send_changes()