from typing import List, Optional
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson


class Timeline(Widget):
    class Routes:
        SEGMENT_SELECTED = "segment_selected_cb"
        CLICK = "click"

    def __init__(
        self,
        frames_count: int,
        intervals: List[List[int]],
        colors: List[List[int]],
        height: Optional[str] = "30px",
        pointer_color: Optional[str] = None,
        tooltip_content: Optional[Widget] = None,
        widget_id: str = None,
    ):
        self._frames_count = frames_count
        self._intervals = intervals
        self._colors = colors
        if len(self._intervals) != len(self._colors):
            raise ValueError("Intervals and colors lists must be of the same length")
        for interval in self._intervals:
            if interval[0] < 0 or interval[1] > self._frames_count:
                raise ValueError("Interval is out of bounds")
        self._height = height
        if pointer_color is None:
            pointer_color = "rgba(151, 151, 151, 1)"
        self._pointer_color = pointer_color
        self._tooltip_content = tooltip_content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "framesCount": self._frames_count,
            "intervals": self._intervals,
            "colors": self._colors,
            "options": {
                "height": self._height,
                "pointerColor": self._pointer_color,
            },
        }

    def get_json_state(self):
        return {
            "pointer": 1,
            "selectedSegment": None,
        }

    def set_pointer(self, pointer):
        StateJson()[self.widget_id]["pointer"] = pointer
        StateJson().send_changes()

    def get_pointer(self):
        return StateJson()[self.widget_id]["pointer"]

    def get_selected_segment(self):
        return StateJson()[self.widget_id]["selectedSegment"]

    def set(self, frames_count, intervals, colors):
        if len(intervals) != len(colors):
            raise ValueError("Intervals and colors lists must be of the same length")
        for interval in intervals:
            if interval[0] < 0 or interval[1] > frames_count:
                raise ValueError("Interval is out of bounds")
        self._frames_count = frames_count
        self._intervals = intervals
        self._colors = colors
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

    def segment_selected(self, func):
        route_path = self.get_route_path(Timeline.Routes.SEGMENT_SELECTED)
        server = self._sly_app.get_server()
        self._segment_selected_handled = True

        @server.post(route_path)
        def _inner():
            selected_segment = StateJson()[self.widget_id]["selectedSegment"]
            func(selected_segment)

        return _inner

    def click(self, func):
        route_path = self.get_route_path(Timeline.Routes.CLICK)
        server = self._sly_app.get_server()
        self._click_handled = True

        @server.post(route_path)
        def _inner():
            pointer = StateJson()[self.widget_id]["pointer"]
            func(pointer)

        return _inner
