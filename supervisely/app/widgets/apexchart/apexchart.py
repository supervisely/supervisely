from typing import Union
from functools import wraps
from supervisely.app.widgets import Widget
from supervisely.app.content import DataJson, StateJson

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

    def click(self, func):
        @self.add_route(self._sly_app.get_server(), Apexchart.Routes.CLICK)
        @wraps(func)
        def wrapped_click(*args, **kwargs):
            # if self.show_loading:
            # self.loading = True
            series_index = StateJson()[self.widget_id]["clicked_value"]["seriesIndex"]
            datapoint_index = StateJson()[self.widget_id]["clicked_value"][
                "dataPointIndex"
            ]
            print(series_index, datapoint_index)
            new_kwargs = dict(
                kwargs,
                serie_index=series_index,
                data_point_index=datapoint_index,
            )
            result = func(*args, **new_kwargs)
            # if self.show_loading:
            # self.loading = False
            return result

        return wrapped_click
