from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import Union, List, Dict, Tuple


class Timeline(Widget):
    class Routes:
        CLICK = "timeline_clicked_cb"

    def __init__(
        self,
        pointer: int,
        frames_count: int,
        intervals: List = [],
        colors: List = [],
        height: int = 30,
        pointer_color: Tuple = (151, 151, 151, 1),
        widget_id: str = None,
    ):

        self._pointer = pointer
        self._frames_count = frames_count
        self._intervals = intervals
        self._colors = colors
        self._height = f"{height}px"
        self._pointer_color = pointer_color
        self._click_handled = True

        if self._max_value is None and len(self._rows_data) != 0:
            check_max_value = []
            for curr_row in self._rows_data:
                check_max_value.append(curr_row["total"])

            self._max_value = max(check_max_value)

        for curr_row in self._rows_data:
            if curr_row.get("segments") is None:
                curr_row["segments"] = {}

        self._content = {
            "maxValue": self._max_value,
            "segments": self._segments,
            "rows": self._rows_data,
        }

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "example1": {
                "content": self._content,
                "options": {
                    "selectable": self._selectable,
                    "collapsable": self._collapsable,
                    "clickableName": self._clickable_name,
                    "clickableSegment": self._clickable_segment,
                    "maxHeight": self._max_height,
                },
                "imageSliderData": self._slider_data,
            }
        }

    def get_json_state(self):
        return {"selectedRows": [], "clickedItem": None}

    # def get_max_value(self):
    #     return DataJson()[self.widget_id]["example1"]["content"]["maxValue"]

    # def set_max_value(self, value: int):
    #     self._max_value = value
    #     DataJson()[self.widget_id]["example1"]["content"]["maxValue"] = self._max_value
    #     DataJson().send_changes()

    # def set_max_height(self, value: int):
    #     self._max_height = f"{value}px"
    #     DataJson()[self.widget_id]["example1"]["options"]["maxHeight"] = self._max_height
    #     DataJson().send_changes()

    # def get_max_height(self):
    #     self._max_height = DataJson()[self.widget_id]["example1"]["options"]["maxHeight"]
    #     return int(self._height[:-2])

    # def disable_selectable(self):
    #     self._selectable = False
    #     DataJson()[self.widget_id]["example1"]["options"]["selectable"] = self._selectable
    #     DataJson().send_changes()

    # def unable_selectable(self):
    #     self._selectable = True
    #     DataJson()[self.widget_id]["example1"]["options"]["selectable"] = self._selectable
    #     DataJson().send_changes()

    # def get_selectable(self):
    #     return DataJson()[self.widget_id]["example1"]["options"]["selectable"]

    # def disable_collapsable(self):
    #     self._collapsable = False
    #     DataJson()[self.widget_id]["example1"]["options"]["collapsable"] = self._collapsable
    #     DataJson().send_changes()

    # def unable_collapsable(self):
    #     self._collapsable = True
    #     DataJson()[self.widget_id]["example1"]["options"]["collapsable"] = self._collapsable
    #     DataJson().send_changes()

    # def get_collapsable(self):
    #     return DataJson()[self.widget_id]["example1"]["options"]["collapsable"]

    # def disable_clickable_name(self):
    #     self._clickable_name = False
    #     DataJson()[self.widget_id]["example1"]["options"]["clickableName"] = self._clickable_name
    #     DataJson().send_changes()

    # def unable_clickable_name(self):
    #     self._clickable_name = True
    #     DataJson()[self.widget_id]["example1"]["options"]["clickableName"] = self._clickable_name
    #     DataJson().send_changes()

    # def get_clickable_name(self):
    #     return DataJson()[self.widget_id]["example1"]["options"]["clickableName"]

    # def disable_clickable_segment(self):
    #     self._clickable_segment = False
    #     DataJson()[self.widget_id]["example1"]["options"][
    #         "clickableSegment"
    #     ] = self._clickable_segment
    #     DataJson().send_changes()

    # def unable_clickable_segment(self):
    #     self._clickable_segment = True
    #     DataJson()[self.widget_id]["example1"]["options"][
    #         "clickableSegment"
    #     ] = self._clickable_segment
    #     DataJson().send_changes()

    # def get_clickable_segment(self):
    #     return DataJson()[self.widget_id]["example1"]["options"]["clickableSegment"]

    # def add_segments(self, segments: List[Dict] = [], send_changes=True):
    #     for curr_segment in segments:
    #         self._segments.append(curr_segment)

    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def get_segments(self):
    #     return DataJson()[self.widget_id]["example1"]["content"]["segments"]

    # def set_segments(self, segments: List[Dict] = [], send_changes=True):
    #     self._content["segments"] = segments
    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def add_rows_data(self, rows_data: List[Dict] = [], send_changes=True):
    #     for curr_rows_data in rows_data:
    #         self._rows_data.append(curr_rows_data)

    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def get_rows_data(self):
    #     return DataJson()[self.widget_id]["example1"]["content"]["rows"]

    # def set_rows_data(self, rows_data: List[Dict] = [], send_changes=True):
    #     self._content["rows"] = rows_data
    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def add_slider_data(self, slider_data: Dict = {}, send_changes=True):
    #     for slider_key, slider_value in slider_data.items():
    #         self._slider_data[slider_key] = slider_value

    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def get_slider_data(self):
    #     return DataJson()[self.widget_id]["example1"]["imageSliderData"]

    # def set_slider_data(self, slider_data: Dict = {}, send_changes=True):
    #     self._slider_data = slider_data
    #     self.update_data()
    #     if send_changes:
    #         DataJson().send_changes()

    # def get_selected_rows(self):
    #     return StateJson()[self.widget_id]["selectedRows"]

    # def click(self, func):
    #     route_path = self.get_route_path(ClassBalance.Routes.CLICK)
    #     server = self._sly_app.get_server()

    #     self._click_handled = True

    #     @server.post(route_path)
    #     def _click():
    #         res = StateJson()[self.widget_id]["clickedItem"]
    #         func(res)

    #     return _click
