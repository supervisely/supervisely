from typing import Union
from functools import wraps
from supervisely.app.widgets.apexchart.apexchart import Apexchart

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

"""
size1 = 10
x1 = list(range(size1))
y1 = np.random.randint(low=10, high=148, size=size1).tolist()
s1 = [{"x": x, "y": y} for x, y in zip(x1, y1)]

size2 = 30
x2 = list(range(size2))
y2 = np.random.randint(low=0, high=300, size=size2).tolist()
s2 = [{"x": x, "y": y} for x, y in zip(x2, y2)]

chart = sly.app.widgets.LineChart(
    title="Max vs Denis",
    series=[{"name": "Max", "data": s1}, {"name": "Denis", "data": s2}],
    xaxis_type="category",
)

@chart.click
def refresh_images_table(datapoint: sly.app.widgets.LineChart.ClickedDataPoint):
    print(f"Line: {datapoint.series_name}")
    print(f"x = {datapoint.x}")
    print(f"y = {datapoint.y}")
"""


class LineChart(Apexchart):
    def __init__(
        self,
        title: str,
        series: list = [],
        zoom: bool = False,
        stroke_curve: Literal["smooth", "straight"] = "smooth",
        stroke_width: int = 2,
        data_labels: bool = False,
        xaxis_type: Literal["numeric", "category", "datetime"] = "numeric",
        xaxis_title: str = None,
        yaxis_title: str = None,
        height: Union[int, str] = 300,
    ):
        self._title = title
        self._series = series
        self._zoom = zoom
        self._stroke_curve = stroke_curve
        self._stroke_width = stroke_width
        self._data_labels = data_labels
        self._xaxis_type = xaxis_type
        self._xaxis_title = xaxis_title
        self._yaxis_title = yaxis_title
        self._widget_height = height

        self._options = {
            "chart": {"type": "line", "zoom": {"enabled": self._zoom}},
            "dataLabels": {"enabled": self._data_labels},
            "stroke": {"curve": self._stroke_curve, "width": self._stroke_width},
            "title": {"text": self._title, "align": "left"},
            "grid": {"row": {"colors": ["#f3f3f3", "transparent"], "opacity": 0.5}},
            "xaxis": {"type": self._xaxis_type},
        }
        if self._xaxis_title is not None:
            self._options["xaxis"]["title"] = {"text": str(self._xaxis_title)}
        if self._yaxis_title is not None:
            self._options["yaxis"] = {"title": {"text": str(self._yaxis_title)}}

        super(LineChart, self).__init__(
            series=self._series,
            options=self._options,
            type="line",
            height=self._widget_height,
        )
