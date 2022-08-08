from __future__ import annotations
from typing import Union, NamedTuple, Any
from functools import wraps
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson

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
        series: list,
        options: dict,
        type: str,
        height: Union[int, str] = "300",
    ):
        self._series = series
        self._options = options
        self._type = type
        self._height = height
        super().__init__(file_path=__file__)

    def get_json_data(self):
        return {
            "series": self._series,
            "options": self._options,
            "type": self._type,
            "height": self._height,
        }

    def get_json_state(self):
        return {"clicked_value": None}

    def get_clicked_value(self):
        return StateJson()[self.widget_id]["clicked_value"]

    def click(self, func):
        @self._sly_app.get_server().post(f"/{self.widget_id}/{Apexchart.Routes.CLICK}")
        def _click():
            value = self.get_clicked_value()
            series_index = value["seriesIndex"]
            series_name = self._series[series_index]["name"]
            data_index = value["dataPointIndex"]
            data = self._series[series_index]["data"][data_index]

            res = Apexchart.ClickedDataPoint(
                series_index, series_name, data_index, data, data["x"], data["y"]
            )
            func(res)

        return _click
