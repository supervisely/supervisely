from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class ClassBalance(Widget):
    """ClassBalance is a widget in Supervisely that allows for displaying input data classes balance on the UI.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/compare-data/classbalance>`_
        (including screenshots and examples).

    :param segments: List of segments to be displayed in the widget.
    :type segments: Optional[List[Dict]]
    :param rows_data: List of rows to be displayed in the widget.
    :type rows_data: Optional[List[Dict]]
    :param slider_data: Dictionary of slider data to be displayed in the widget.
    :type slider_data: Optional[Dict[str, List]]
    :param max_value: Maximum value of the widget.
    :type max_value: Optional[int]
    :param max_height: Maximum height of the widget in pixels.
    :type max_height: Optional[int]
    :param rows_height: Height of the rows in pixels.
    :type rows_height: Optional[int]
    :param selectable: If True, the widget will be selectable.
    :type selectable: Optional[bool]
    :param collapsable: If True, the widget will be collapsable.
    :type collapsable: Optional[bool]
    :param clickable_name: If True, the widget will be clickable by name.
    :type clickable_name: Optional[bool]
    :param clickable_segment: If True, the widget will be clickable by segment.
    :type clickable_segment: Optional[bool]
    :param widget_id: Unique widget identifier.
    :type widget_id: str

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import ClassBalance

        max_value = 1000
        segments = [
            {"name": "train", "key": "train", "color": "#1892f8"},
            {"name": "val", "key": "val", "color": "#25e298"},
            {"name": "test", "key": "test", "color": "#fcaf33"},
        ]

        rows_data = [
            {
                "nameHtml": "<strong>black-pawn</strong>",
                "name": "black-pawn",
                "total": 1000,
                "disabled": False,
                "segments": {"train": 600, "val": 350, "test": 50},
            },
            {
                "nameHtml": "<strong>white-pawn</strong>",
                "name": "white-pawn",
                "total": 700,
                "disabled": False,
                "segments": {"train": 400, "val": 250, "test": 50},
            },
        ]

        slider_data = {
            "black-pawn": [
                {
                    "moreExamples": ["https://www.w3schools.com/howto/img_nature.jpg"],
                    "preview": "https://www.w3schools.com/howto/img_nature.jpg",
                }
            ],
            "white-pawn": [
                {
                    "moreExamples": ["https://i.imgur.com/35pUPD2.jpg"],
                    "preview": "https://i.imgur.com/35pUPD2.jpg",
                }
            ],
        }

        class_balance_1 = ClassBalance(
            max_value=max_value,
            segments=segments,
            rows_data=rows_data,
            slider_data=slider_data,
            max_height=700,
            collapsable=True,
        )
    """

    class Routes:
        CLICK = "class_balance_clicked_cb"

    def __init__(
        self,
        segments: Optional[List[Dict]] = [],
        rows_data: Optional[List[Dict]] = [],
        slider_data: Optional[Dict[str, List]] = {},
        max_value: Optional[int] = None,
        max_height: Optional[int] = 350,
        rows_height: Optional[int] = 100,
        selectable: Optional[bool] = True,
        collapsable: Optional[bool] = False,
        clickable_name: Optional[bool] = False,
        clickable_segment: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._segments = segments
        self._rows_data = rows_data
        self._slider_data = slider_data
        self._max_value = max_value
        self._max_height = f"{max_height}px"
        self._rows_height = f"{rows_height}px"
        self._selectable = selectable
        self._collapsable = collapsable
        self._clickable_name = clickable_name
        self._clickable_segment = clickable_segment
        self._click_handled = False

        if self._max_value is None:
            if len(self._rows_data) != 0:
                self._max_value = max([curr_row["total"] for curr_row in self._rows_data])
            else:
                self._max_value = 0

        for curr_row in self._rows_data:
            if curr_row.get("segments") is None:
                curr_row["segments"] = {}

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - content: Dictionary with the following fields:
                - maxValue: Maximum value of the widget.
                - segments: List of segments to be displayed in the widget.
                - rows: List of rows to be displayed in the widget.
            - options: Dictionary with the following fields:
                - selectable: If True, the widget will be selectable.
                - collapsable: If True, the widget will be collapsable.
                - clickableName: If True, the widget will be clickable by name.
                - clickableSegment: If True, the widget will be clickable by segment.
                - maxHeight: Maximum height of the widget in pixels.
            - imageSliderData: Dictionary with the following fields:
                - name: Name of the row.
                - preview: Preview image.
                - moreExamples: List of preview images.
            - imageSliderOptions: Dictionary with the following fields:
                - selectable: If True, the widget will be selectable.
                - height: Height of the rows in pixels.

        :return: Dictionary with widget data.
        :rtype: Dict[str, Any]
        """
        return {
            "content": {
                "maxValue": self._max_value,
                "segments": deepcopy(self._segments),
                "rows": deepcopy(self._rows_data),
            },
            "options": {
                "selectable": self._selectable,
                "collapsable": self._collapsable,
                "clickableName": self._clickable_name,
                "clickableSegment": self._clickable_segment,
                "maxHeight": self._max_height,
            },
            "imageSliderData": deepcopy(self._slider_data),
            "imageSliderOptions": {"selectable": False, "height": self._rows_height},
        }

    def get_json_state(self) -> Dict[str, Any]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - selectedRows: List of selected rows.
            - clickedItem: Name of the clicked item.

        :return: Dictionary with widget state.
        :rtype: Dict[str, Any]
        """
        return {"selectedRows": [], "clickedItem": None}

    @property
    def is_selectable(self) -> bool:
        """Returns True if the widget is selectable, False otherwise.

        :return: True if the widget is selectable, False otherwise.
        :rtype: bool
        """
        self._selectable = DataJson()[self.widget_id]["options"]["selectable"]
        return self._selectable

    @property
    def is_collapsable(self) -> bool:
        """Returns True if the widget is collapsable, False otherwise.

        :return: True if the widget is collapsable, False otherwise.
        :rtype: bool
        """
        self._collapsable = DataJson()[self.widget_id]["options"]["collapsable"]
        return self._collapsable

    @property
    def is_clickable_name(self) -> bool:
        """Returns True if the widget is clickable by name, False otherwise.

        :return: True if the widget is clickable by name, False otherwise.
        :rtype: bool
        """
        self._clickable_name = DataJson()[self.widget_id]["options"]["clickableName"]
        return self._clickable_name

    @property
    def is_clickable_segment(self) -> bool:
        """Returns True if the widget is clickable by segment, False otherwise.

        :return: True if the widget is clickable by segment, False otherwise.
        :rtype: bool
        """
        self._clickable_segment = DataJson()[self.widget_id]["options"]["clickableSegment"]
        return self._clickable_segment

    def get_max_value(self) -> int:
        """Returns maximum value of the widget.

        :return: Maximum value of the widget.
        :rtype: int
        """
        self._max_value = DataJson()[self.widget_id]["content"]["maxValue"]
        return self._max_value

    def set_max_value(self, value: int) -> None:
        """Sets maximum value of the widget.

        :param value: Maximum value of the widget.
        :type value: int
        """
        self._max_value = value
        DataJson()[self.widget_id]["content"]["maxValue"] = self._max_value
        DataJson().send_changes()

    def set_max_height(self, value: int) -> None:
        """Sets maximum height of the widget.

        :param value: Maximum height of the widget in pixels.
        :type value: int
        :raises TypeError: If max height value is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError("Max height must be an integer.")
        self._max_height = f"{value}px"
        DataJson()[self.widget_id]["options"]["maxHeight"] = self._max_height
        DataJson().send_changes()

    def get_max_height(self) -> int:
        """Returns maximum height of the widget.

        :return: Maximum height of the widget in pixels.
        :rtype: int
        """
        self._max_height = DataJson()[self.widget_id]["options"]["maxHeight"]
        return int(self._max_height[:-2])

    def add_segments(self, segments: Optional[List[Dict]] = []) -> None:
        """Appends list of segments to the existing segments.

        This method will not overwrite the existing segments, but append to it.
        To overwrite segments, use :meth:`set_segments`.

        :param segments: List of segments to be displayed in the widget.
        :type segments: Optional[List[Dict]]
        :raises ValueError: If segment name or key already exists.
        """
        for curr_segment in segments:
            if curr_segment["name"] in [seg["name"] for seg in self._segments]:
                raise ValueError("Segment name already exists.")
            if curr_segment["key"] in [seg["key"] for seg in self._segments]:
                raise ValueError("Segment key already exists.")
            self._segments.append(curr_segment)

        self.update_data()
        DataJson().send_changes()

    def get_segments(self) -> List[Dict]:
        """Returns list of segments to be displayed in the widget.

        :return: List of segments to be displayed in the widget.
        :rtype: List[Dict]
        """
        self._segments = DataJson()[self.widget_id]["content"]["segments"]
        return self._segments

    def set_segments(self, segments: Optional[List[Dict]] = []) -> None:
        """Sets list of segments to be displayed in the widget.

        This method will overwrite the existing segments, not append to it.
        To append segments, use :meth:`add_segments`.

        :param segments: List of segments to be displayed in the widget.
        :type segments: Optional[List[Dict]]
        """
        self._segments = segments
        self.update_data()
        DataJson().send_changes()

    def add_rows_data(self, rows_data: Optional[List[Dict]] = []) -> None:
        """Appends list of rows to the existing rows.

        This method will not overwrite the existing rows, but append to it.
        To overwrite rows, use :meth:`set_rows_data`.

        :param rows_data: List of rows to be displayed in the widget.
        :type rows_data: Optional[List[Dict]]
        :raises ValueError: If row data with the same name already exists.
        """
        for curr_rows_data in rows_data:
            if curr_rows_data["name"] in [row_data["name"] for row_data in self._rows_data]:
                raise ValueError("Row data with the same name already exists.")
            self._rows_data.append(curr_rows_data)

        self.update_data()
        DataJson().send_changes()

    def get_rows_data(self) -> List[Dict]:
        """Returns list of rows to be displayed in the widget.

        :return: List of rows to be displayed in the widget.
        :rtype: List[Dict]
        """
        self._rows_data = DataJson()[self.widget_id]["content"]["rows"]
        return self._rows_data

    def set_rows_data(self, rows_data: Optional[List[Dict]] = []) -> None:
        """Sets list of rows to be displayed in the widget.

        This method will overwrite the existing rows, not append to it.
        To append rows, use :meth:`add_rows_data`.

        :param rows_data: List of rows to be displayed in the widget.
        :type rows_data: Optional[List[Dict]]
        """
        self._rows_data = rows_data
        self.update_data()
        DataJson().send_changes()

    def add_slider_data(self, slider_data: Optional[Dict] = {}) -> None:
        """Appends dictionary of slider data to the existing slider data.

        This method will not overwrite the existing slider data, but append to it.
        To overwrite slider data, use :meth:`set_slider_data`.

        :param slider_data: Dictionary of slider data to be displayed in the widget.
        :type slider_data: Optional[Dict[str, List]]
        :raises ValueError: If row data with the same name does not exist.
        """
        for slider_key, slider_value in slider_data.items():
            if slider_key not in [row_data["name"] for row_data in self._rows_data]:
                raise ValueError("Row data with the same name does not exist.")
            self._slider_data[slider_key] = slider_value

        self.update_data()
        DataJson().send_changes()

    def get_slider_data(self) -> Dict[str, List]:
        """Returns dictionary of slider data to be displayed in the widget.

        :return: Dictionary of slider data to be displayed in the widget.
        :rtype: Dict[str, List]
        """
        self._slider_data = DataJson()[self.widget_id]["imageSliderData"]
        return self._slider_data

    def set_slider_data(self, slider_data: Optional[Dict[str, List]] = {}):
        """Sets dictionary of slider data to be displayed in the widget.

        This method will overwrite the existing slider data, not append to it.
        To append slider data, use :meth:`add_slider_data`.

        :param slider_data: Dictionary of slider data to be displayed in the widget.
        :type slider_data: Optional[Dict[str, List]]
        :raises ValueError: If row data with the same name does not exist.
        """
        for slider_key in slider_data.keys():
            if slider_key not in [row_data["name"] for row_data in self._rows_data]:
                raise ValueError("Row data with the same name does not exist.")
        self._slider_data = slider_data
        self.update_data()
        DataJson().send_changes()

    def get_selected_rows(self) -> List[str]:
        """Returns list of selected rows.

        :return: List of selected rows.
        :rtype: List[str]
        """
        return StateJson()[self.widget_id]["selectedRows"]

    def click(self, func: Callable[[str], None]) -> Callable[[], None]:
        """Decorator for the function to be called when the widget is clicked.

        :param func: Function to be called when the widget is clicked.
        :type func: Callable[[str], None]
        :return: Decorated function.
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(ClassBalance.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = StateJson()[self.widget_id]["clickedItem"]
            func(res)

        return _click
