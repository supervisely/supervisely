from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List


class Timeline(Widget):
    class Routes:
        CLICK = "timeline_clicked_cb"

    def __init__(
        self,
        pointer: int = 0,
        frames_count: int = 0,
        intervals: List = [],
        colors: List = [],
        height: int = 30,
        pointer_color: str = "",
        widget_id: str = None,
    ):

        self._pointer = pointer
        self._frames_count = frames_count
        self._intervals = intervals
        self._colors = colors
        self._height = f"{height}px"
        self._pointer_color = pointer_color
        self._selected_segment = None
        self._click_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "framesCount": self._frames_count,
            "intervals": self._intervals,
            "colors": self._colors,
            "options": {"height": self._height, "pointerColor": self._pointer_color},
        }

    def get_json_state(self):
        return {"pointer": self._pointer, "selectedSegment": self._selected_segment}

    def set_pointer(self, value: int):
        self._pointer = value
        StateJson()[self.widget_id]["pointer"] = self._pointer
        StateJson().send_changes()

    def get_pointer(self):
        return StateJson()[self.widget_id]["pointer"]

    def set_frames_count(self, value: int):
        self._frames_count = value
        DataJson()[self.widget_id]["framesCount"] = self._frames_count
        DataJson().send_changes()

    def get_frames_count(self):
        return DataJson()[self.widget_id]["framesCount"]

    def add_intervals(self, intervals: List = [], colors: List = []):
        self._intervals.extend(intervals)
        self._colors.extend(colors)
        if self._intervals[-1][1] > self._frames_count:
            self._frames_count = self._intervals[-1][1]
            DataJson()[self.widget_id]["framesCount"] = self._frames_count
        DataJson()[self.widget_id]["intervals"] = self._intervals
        DataJson()[self.widget_id]["colors"] = self._colors
        DataJson().send_changes()

    def set_intervals(self, intervals: List = [], colors: List = []):
        self._intervals = intervals
        self._colors = colors
        self._frames_count = self._intervals[-1][1]
        DataJson()[self.widget_id]["framesCount"] = self._frames_count
        DataJson()[self.widget_id]["intervals"] = self._intervals
        DataJson()[self.widget_id]["colors"] = self._colors
        DataJson().send_changes()

    def get_intervals(self):
        return DataJson()[self.widget_id]["intervals"]

    def get_colors(self):
        return DataJson()[self.widget_id]["colors"]

    def set_height(self, value: int):
        self._height = f"{value}px"
        DataJson()[self.widget_id]["options"]["height"] = self._height
        DataJson().send_changes()

    def get_height(self):
        return DataJson()[self.widget_id]["options"]["height"]

    def set_pointer_color(self, value: str):
        self._pointer_color = value
        DataJson()[self.widget_id]["options"]["pointerColor"] = self._pointer_color
        DataJson().send_changes()

    def get_pointer_color(self):
        return DataJson()[self.widget_id]["options"]["pointerColor"]

    def get_select_segment(self):
        return StateJson()[self.widget_id]["selectedSegment"]

    def click(self, func):
        route_path = self.get_route_path(Timeline.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = StateJson()[self.widget_id]["selectedSegment"]
            func(res)

        return _click
