from typing import List
from supervisely.app.widgets import Widget, Container, generate_id, LinePlot
from supervisely.sly_logger import logger
from supervisely._utils import batched, rand_str
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.content import StateJson, DataJson


class GridPlot(Widget):
    def __init__(
        self,
        data: list[dict] = [],
        columns: int = 1,
        gap: int = 10,
        widget_id: str = None,
    ):  
        self._widgets = {}
        self._columns = columns
        self._gap = gap

        for plot_data in data:
            self._widgets[plot_data['title']] = LinePlot(title=plot_data['title'], series=plot_data['series'])

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

    def add_scalar(self, identifier: str, y, x):
        plot_title, series_name = identifier.split('/')
        self._widgets[plot_title].add_to_series(name_or_id=series_name, data=[{"x": x, "y": y}])
    
    def add_scalars(self, plot_title: str, new_values: dict, x):
        for series_name in new_values.keys():
            self._widgets[plot_title].add_to_series(name_or_id=series_name, data=[{"x": x, "y": new_values[series_name]}])