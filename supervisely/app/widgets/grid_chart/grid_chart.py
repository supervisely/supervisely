from functools import wraps
from typing import Dict, List, Union

from supervisely._utils import batched, rand_str
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets.apexchart.apexchart import Apexchart
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.widgets.line_chart.line_chart import LineChart
from supervisely.app.widgets.line_plot.line_plot import LinePlot
from supervisely.app.widgets.widget import Widget, generate_id
from supervisely.sly_logger import logger

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


class GridChart(Widget):
    def __init__(
        self,
        data: List[Dict or str] = [],
        columns: int = 1,
        gap: int = 10,
        widget_id: str = None,
    ):
        # TODO expand LineChart -> ApexChart
        self._widgets = {}
        self._columns = columns
        self._gap = gap

        for plot_data in data:
            if isinstance(plot_data, dict):
                # self._widgets[plot_data['title']] = LinePlot(title=plot_data['title'], series=plot_data.get('series', []), show_legend=plot_data.get('show_legend', True))
                # passing parameters in this way will eventually result in a JsonPatchConflict error
                plot_data
                self._widgets[plot_data["title"]] = LineChart(**plot_data)
            else:
                self._widgets[plot_data] = LineChart(title=plot_data, series=[])

        if self._columns < 1:
            raise ValueError(f"columns ({self._columns}) < 1")
        if self._columns > len(self._widgets):
            logger.warn(
                f"Number of columns ({self._columns}) > number of widgets ({len(self._widgets)}). Columns are set to {len(self._widgets)}"
            )
            self._columns = len(self._widgets)

        self._content = None
        if self._columns == 1:
            self._content = Container(
                direction="vertical",
                widgets=self._widgets.values(),
                gap=self._gap,
                widget_id=generate_id(),
            )
        else:
            rows = []
            num_empty = len(self._widgets) % self._columns
            for i in range(num_empty):
                self._widgets[generate_id()] = Empty()
            for batch in batched(list(self._widgets.values()), batch_size=self._columns):
                rows.append(
                    Container(
                        direction="horizontal",
                        widgets=batch,
                        gap=self._gap,
                        fractions=[1] * len(batch),
                        widget_id=generate_id(),
                        overflow=None,
                        widgets_style="flex-direction: column",
                    )
                )
            self._content = Container(
                direction="vertical",
                widgets=rows,
                gap=self._gap,
                widget_id=generate_id(),
            )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None

    def set_series(self, series: list, send_changes=True):
        return

    def set_colors(self, colors: Dict[str, List[str]], send_changes=True):
        """Set colors for every line in the grid chart.

        Args:
            colors (Dict[str, List[str]]): Set new colors for every line as a string, rgb, or HEX. Example: `{'title':['red', 'rgb(0,255,0), '#0000FF']}`
            send_changes (bool, optional): Send changes to the chart. Defaults to True.
        """
        for title, clrs in colors.items():
            widget: Apexchart = self._widgets[title]
            widget.set_colors(colors=clrs, send_changes=send_changes)
