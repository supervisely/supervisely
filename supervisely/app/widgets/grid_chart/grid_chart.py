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


class GridChart(Widget):
    def __init__(
        self,
        data: List[Union[dict, str]] = [],
        columns: int = 1,
        gap: int = 10,
        widget_id: str = None,
    ):
        # TODO maybe generalize LineChart -> ApexChart ???
        self._widgets = {}
        self._columns = columns
        self._gap = gap

        if len(data) > 0:  # TODO: add the self.add_data([data:dict]) method
            for plot_data in data:
                if isinstance(plot_data, dict):
                    self._widgets[plot_data["title"]] = LineChart(**plot_data)
                else:
                    self._widgets[plot_data] = LineChart(title=plot_data, series=[])

            if self._columns < 1:
                raise ValueError(f"columns ({self._columns}) < 1")
            elif self._columns > len(self._widgets):
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

    def add_scalar(
        self,
        identifier: str,
        y: Union[float, int],
        x: Union[float, int],
    ) -> None:
        """
        Add scalar to a single series on plot. If no series with name,
         defined in `identifier` exists, one will be created automatically.

        :param identifier: '/'-separated plot and series name
        :type identifier: str
        :param y: y value
        :type y: float | int
        :param x: x value
        :type x: float | int
        """
        plot_title, series_name = identifier.split("/")  # TODO: maybe Dict[str,str] is way better??
        w: LineChart = self._widgets[plot_title]
        series_id, series = w.get_series_by_name(series_name)

        if series is not None:
            w.add_to_series(name_or_id=series_id, data=[(x, y)])
        else:
            w.add_series(name=series_name, x=[x], y=[y])

    def add_scalars(
        self,
        plot_title: str,
        new_values: Dict[str, Union[float, int]],
        x: Union[float, int],
    ):
        """
        Add scalars to several series on one plot at point `x`

        :param plot_title: name of existing plot
        :type plot_title: str
        :param new_values: A dictionary in the `{'series_1': y1, 'series_2': y2, ...}` format.
        :type new_values: Dict[str, (float | int)]]
        :param x: value of `x`
        :type x: float | int
        """
        w: LineChart = self._widgets[plot_title]
        for s_name in new_values.keys():
            series_id, series = w.get_series_by_name(s_name)
            if series is not None:
                w.add_to_series(
                    name_or_id=series_id,
                    data=w._wrap_xy(x, new_values[s_name]),
                )
            else:
                w.add_series(name=s_name, x=[x], y=[new_values[s_name]])
