from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Dict


class ClassBalance(Widget):
    class Routes:
        CLICK = "class_balance_clicked_cb"

    def __init__(
        self,
        max_value: int = None,
        segments: List[Dict] = [],
        rows_data: List[Dict] = [],
        slider_data: Dict = {},
        max_height: int = 300,
        selectable: bool = True,
        collapsable: bool = False,
        clickable_name: bool = False,
        clickable_segment: bool = False,
        widget_id: str = None,
    ):
        self._max_value = max_value
        self._segments = segments
        self._rows_data = rows_data
        self._slider_data = slider_data
        self._max_height = f"{max_height}px"
        self._selectable = selectable
        self._collapsable = collapsable
        self._clickable_name = clickable_name
        self._clickable_segment = clickable_segment
        self._click_handled = False

        if self._max_value is None and len(self._rows_data) != 0:
            check_max_value = []
            for curr_row in self._rows_data:
                check_max_value.append(curr_row["total"])

            self._max_value = max(check_max_value)

        for curr_row in self._rows_data:
            if curr_row.get("segments") is None:
                curr_row["segments"] = {}

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "content": {
                "maxValue": self._max_value,
                "segments": self._segments,
                "rows": self._rows_data,
            },
            "options": {
                "selectable": self._selectable,
                "collapsable": self._collapsable,
                "clickableName": self._clickable_name,
                "clickableSegment": self._clickable_segment,
                "maxHeight": self._max_height,
            },
            "imageSliderData": self._slider_data,
        }

    def get_json_state(self):
        return {"selectedRows": [], "clickedItem": None}

    @property
    def is_selectable(self):
        self._selectable = DataJson()[self.widget_id]["options"]["selectable"]
        return self._selectable

    @property
    def is_collapsable(self):
        self._collapsable = DataJson()[self.widget_id]["options"]["collapsable"]
        return self._collapsable

    @property
    def is_clickable_name(self):
        self._clickable_name = DataJson()[self.widget_id]["options"]["clickableName"]
        return self._clickable_name

    @property
    def is_clickable_segment(self):
        self._clickable_segment = DataJson()[self.widget_id]["options"]["clickableSegment"]
        return self._clickable_segment

    def get_max_value(self):
        self._max_value = DataJson()[self.widget_id]["content"]["maxValue"]
        return self._max_value

    def set_max_value(self, value: int):
        self._max_value = value
        DataJson()[self.widget_id]["content"]["maxValue"] = self._max_value
        DataJson().send_changes()

    def set_max_height(self, value: int):
        self._max_height = f"{value}px"
        DataJson()[self.widget_id]["options"]["maxHeight"] = self._max_height
        DataJson().send_changes()

    def get_max_height(self):
        self._max_height = DataJson()[self.widget_id]["options"]["maxHeight"]
        return int(self._height[:-2])

    def add_segments(self, segments: List[Dict] = [], send_changes=True):
        for curr_segment in segments:
            if curr_segment["name"] in [seg["name"] for seg in self._segments]:
                raise ValueError("Segment name already exists.")
            if curr_segment["key"] in [seg["key"] for seg in self._segments]:
                raise ValueError("Segment key already exists.")
            self._segments.append(curr_segment)

        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def get_segments(self):
        self._segments = DataJson()[self.widget_id]["content"]["segments"]
        return self._segments

    def set_segments(self, segments: List[Dict] = [], send_changes=True):
        self._segments = segments
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def add_rows_data(self, rows_data: List[Dict] = [], send_changes=True):
        for curr_rows_data in rows_data:
            if curr_rows_data["name"] in [row_data["name"] for row_data in self._rows_data]:
                raise ValueError("Row data with the same name already exists.")
            self._rows_data.append(curr_rows_data)

        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def get_rows_data(self):
        self._rows_data = DataJson()[self.widget_id]["content"]["rows"]
        return self._rows_data

    def set_rows_data(self, rows_data: List[Dict] = [], send_changes=True):
        self._rows_data = rows_data
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def add_slider_data(self, slider_data: Dict = {}, send_changes=True):
        for slider_key, slider_value in slider_data.items():
            if slider_key not in [row_data["name"] for row_data in self._rows_data]:
                raise ValueError("Row data with the same name does not exist.")
            self._slider_data[slider_key] = slider_value

        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def get_slider_data(self):
        self._slider_data = DataJson()[self.widget_id]["imageSliderData"]
        return self._slider_data

    def set_slider_data(self, slider_data: Dict = {}, send_changes=True):
        self._slider_data = slider_data
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def get_selected_rows(self):
        return StateJson()[self.widget_id]["selectedRows"]

    def click(self, func):
        route_path = self.get_route_path(ClassBalance.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = StateJson()[self.widget_id]["clickedItem"]
            func(res)

        return _click
