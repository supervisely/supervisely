from __future__ import annotations

import traceback
from functools import wraps
from typing import Any, List, NamedTuple, Union, Tuple, Literal

from supervisely import logger
from supervisely.app.content import DataJson, StateJson
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.widgets import Widget

NumT = Union[int, float]
"""
size1 = 10
x1 = list(range(size1))
y1 = np.random.randint(low=10, high=148, size=size1).tolist()
s1 = [{"x": x, "y": y} for x, y in zip(x1, y1)]

size2 = 30
x2 = list(range(size2))
y2 = np.random.randint(low=0, high=300, size=size2).tolist()
s2 = [{"x": x, "y": y} for x, y in zip(x2, y2)]

chart = sly.app.widgets.Apexchart(
    series=[{"name": "Max", "data": s1}, {"name": "Denis", "data": s2}],
    options={
        "chart": {"type": "line", "zoom": {"enabled": False}},
        "dataLabels": {"enabled": False},
        # "stroke": {"curve": "straight"},
        "stroke": {"curve": "smooth", "width": 2},
        "title": {"text": "Product Trends by Month", "align": "left"},
        "grid": {"row": {"colors": ["#f3f3f3", "transparent"], "opacity": 0.5}},
        "xaxis": {"type": "category"},
    },
    type="line",
)
"""


class Apexchart(Widget):
    class Routes:
        CLICK = "chart_clicked_cb"

    class ClickedDataPoint(NamedTuple):
        series_index: int
        series_name: str
        data_index: int
        data: dict
        x: Any
        y: Any

    def __init__(
        self,
        series: List[dict],
        options: dict,
        type: str,
        height: Union[int, str] = "300",
        sly_options: dict = {},
    ):
        self._series = series
        self._options = options
        self._type = type
        self._height = height
        self._click_handled = False
        self._sly_options = sly_options
        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {
            "series": self._series,
            "options": self._options,
            "type": self._type,
            "height": self._height,
            "sly_options": self._sly_options,
        }

    def get_json_state(self):
        return {"clicked_value": None}

    def get_clicked_value(self):
        return StateJson()[self.widget_id]["clicked_value"]

    def get_clicked_datapoint(self) -> Apexchart.ClickedDataPoint:
        value = self.get_clicked_value()
        series_index = value["seriesIndex"]
        data_index = value["dataPointIndex"]
        if series_index == -1 and data_index != -1:
            # zero point (0,0) click
            series_index = 0
        if series_index == -1 or data_index == -1:
            return
        series_name = self._series[series_index]["name"]
        data = self._series[series_index]["data"][data_index]
        res = Apexchart.ClickedDataPoint(
            series_index, series_name, data_index, data, data["x"], data["y"]
        )
        return res

    def click(self, func):
        route_path = self.get_route_path(Apexchart.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_clicked_datapoint()
            if res is not None:
                try:
                    return func(res)
                except Exception as e:
                    logger.error(traceback.format_exc(), exc_info=True, extra={"exc_str": str(e)})
                    raise e

        return _click

    def add_series(self, name: str, x: list, y: list, send_changes=True):
        if len(x) != len(y):
            raise ValueError(f"Lists x and y have different lenght, {len(x)} != {len(y)}")

        dtype = self._sly_options.get("data_type", "dict")
        data = [{"x": px, "y": py} if dtype == "dict" else (px, py) for px, py in zip(x, y)]

        series = {"name": name, "data": data}
        self._series.append(series)
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def set_title(self, title: str, send_changes=True):
        if self._options.get("title") is None:
            self._options["title"] = {}
        self._options["title"]["text"] = title
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def set_series(self, series: list, send_changes=True):
        self._series = series
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def set_colors(self, colors: list, send_changes=True):
        """
        Set colors for every series in the chart.

        :param colors: Set new colors for every series as a string, rgb, or HEX. Example: `{'title' : ['red', 'rgb(0,255,0), '#0000FF']}`.
        :type colors: Dict[str, List[str]]
        :param send_changes: Send changes to the chart. Defaults to True.
        :type send_changes: bool, optional
        """
        self._options["colors"] = colors
        self.update_data()
        if send_changes:
            DataJson().send_changes()

    def add_to_series(
        self,
        name_or_id: Union[str, int],
        data: Union[List[tuple], List[dict], tuple, dict],
        send_changes=True,
    ) -> None:
        if isinstance(name_or_id, int):
            series_id = name_or_id
        else:
            series_id, _ = self.get_series_by_name(name_or_id)

        if isinstance(data, List):
            data_list = self._list_of_point_dicts_to_list_of_tuples(data)
        else:
            # single datapoint
            data_list = self._list_of_point_dicts_to_list_of_tuples([data])

        self._series[series_id]["data"].extend(data_list)
        DataJson()[self.widget_id]["series"] = self._series

        if send_changes:
            DataJson().send_changes()
            self._update_series_in_apexhart()

    def get_series_by_name(self, name):
        series_list = DataJson()[self.widget_id]["series"]
        series_id, series_data = next(
            ((i, series) for i, series in enumerate(series_list) if series["name"] == name),
            (None, None),
        )
        # assert series_id is not None, KeyError("Series with name: {name} doesn't exists.")
        return series_id, series_data

    def _update_series_in_apexhart(self) -> None:
        # necessary for the force rendering of data
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"sly-apexchart-{self.widget_id}.update-series",
                        "payload": {},
                    }
                }
            )
        )

    def _list_of_point_dicts_to_list_of_tuples(
        self, point_dcts: List[Union[dict, tuple]]
    ) -> List[Tuple[NumT, NumT]]:
        if len(point_dcts) == 0:
            return []
        if isinstance(point_dcts[0], tuple):
            return point_dcts
        return [self._point_dict_to_tuple(dct) for dct in point_dcts]

    def _point_dict_to_tuple(self, point_dct: dict):
        return (point_dct["x"], point_dct["y"])
