from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, List, Literal, Optional
from uuid import uuid4

from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.io.env import task_id as env_task_id


class DebouncedEventHandler:
    def __init__(self, debounce_time: float = 0.1):
        self._event_queue = []
        self._debounce_time = debounce_time
        self._task = None

    async def _process_events(self, func: Callable):
        await asyncio.sleep(self._debounce_time)
        aggregated_events = self._event_queue.copy()
        self._event_queue.clear()
        func(aggregated_events)

    def handle_event(self, event_data, func: Callable):
        self._event_queue.append(event_data)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_events(func))


class SelectedIds(BaseModel):
    selected_ids: List[int]


class Bokeh(Widget):
    """
    Bokeh widget for creating interactive plots.

    Note:
        Only Bokeh version 3.1.1 is supported.

    :param plots: List of plots to be displayed.
    :type plots: List[Plot]
    :param width: Width of the chart in pixels.
    :type width: int
    :param height: Height of the chart in pixels.
    :type height: int
    :param tools: List of tools to be displayed on the chart.
    :type tools: List[str]
    :param toolbar_location: Location of the toolbar.
    :type toolbar_location: Literal["above", "below", "left", "right"]
    :param x_axis_visible: If True, x-axis will be visible.
    :type x_axis_visible: bool
    :param y_axis_visible: If True, y-axis will be visible.
    :type y_axis_visible: bool
    :param grid_visible: If True, grid will be visible.
    :type grid_visible: bool
    :param widget_id: Unique widget identifier.
    :type widget_id: str
    :param show_legend: If True, ckickable legend widget will be displayed.
    :type show_legend: bool
    :param legend_location: Location of the clickable legend widget.
    :type legend_location: Literal["left", "top", "right", "bottom"]
    :param legend_click_policy: Click policy of the clickable legend widget.
    :type legend_click_policy: Literal["hide", "mute"]

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Bokeh, IFrame

        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [1, 2, 3, 4, 5],
            "radius": [10, 20, 30, 40, 50],
            "colors": ["red", "green", "blue", "yellow", "purple"],
            "ids": [1, 2, 3, 4, 5],
            "names": ["kiwi", "kiwi", "lemon", "lemon", "lemon"],
        }

        plot_lemon = Bokeh.Circle(name="lemon")
        plot_kiwi = Bokeh.Circle(name="kiwi")
        bokeh = Bokeh(
            x_axis_visible=True,
            y_axis_visible=True,
            grid_visible=True,
            show_legend=True,
        )
        bokeh.add_data(**data)
        bokeh.add_plots([plot_lemon, plot_kiwi])

        # To allow the widget to be interacted with, you need to add it to the IFrame widget.
        iframe = IFrame()
        iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
    """

    class Routes:
        VALUE_CHANGED = "value_changed"
        HTML_ROUTE = "bokeh.html"

    class Plot(ABC):
        def __init__(self, name: Optional[str] = None, **kwargs):

            self._name = name or str(uuid4())
            self.kwargs = kwargs

        @abstractmethod
        def add(self, plot, source) -> None:
            pass

        @property
        def name(self) -> str:
            return self._name

    class Circle(Plot):
        def add(self, plot, source) -> None:
            from bokeh.models import (  # pylint: disable=import-error
                CDSView,
                GroupFilter,
            )

            filters = [GroupFilter(column_name="names", group=self.name, name=self.name)]
            view = CDSView(source=source, filters=filters)
            return plot.circle(
                "x",
                "y",
                radius="radius",
                fill_color="colors",
                fill_alpha=0.5,
                source=source,
                line_color=None,
                view=view,
            )

    class Scatter(Plot):
        def add(self, plot, source) -> None:
            from bokeh.models import (  # pylint: disable=import-error
                CDSView,
                GroupFilter,
            )

            filters = [GroupFilter(column_name="names", group=self.name, name=self.name)]
            view = CDSView(source=source, filters=filters)
            return plot.scatter(
                "x",
                "y",
                size="radius",
                color="colors",
                fill_alpha=0.5,
                source=source,
                view=view,
            )

    class Line(Plot):
        def add(self, plot, source) -> None:
            from bokeh.models import (  # pylint: disable=import-error
                CDSView,
                GroupFilter,
            )

            filters = [GroupFilter(column_name="names", group=self.name, name=self.name)]
            view = CDSView(source=source, filters=filters)
            return plot.line("x", "y", source=source, view=view, line_width=2)

    def __init__(
        self,
        plots: List[Plot] = None,
        width: int = 1000,
        height: int = 600,
        tools: List[str] = [
            "pan",
            "wheel_zoom",
            "box_zoom",
            "reset",
            # "save",
            "poly_select",
            "tap",
            "lasso_select",
        ],
        toolbar_location: Literal["above", "below", "left", "right"] = "above",
        x_axis_visible: bool = False,
        y_axis_visible: bool = False,
        grid_visible: bool = False,
        widget_id: Optional[str] = None,
        show_legend: bool = False,
        legend_location: Literal["left", "top", "right", "bottom"] = "right",
        legend_click_policy: Literal["hide", "mute"] = "hide",
        **kwargs,
    ):
        import bokeh  # pylint: disable=import-error

        # check Bokeh version compatibility (only 3.1.1 is supported)
        if bokeh.__version__ != "3.1.1":
            raise RuntimeError(f"Bokeh version {bokeh.__version__} is not supported. Use 3.1.1")

        self._source_data = {"x": [], "y": [], "radius": [], "colors": [], "ids": [], "names": []}
        self._source = None
        self.widget_id = widget_id
        self._plots = plots or []
        self._view = None

        self._width = width
        self._height = height
        self._tools = tools
        self._toolbar_location = toolbar_location
        self._x_axis_visible = x_axis_visible
        self._y_axis_visible = y_axis_visible
        self._grid_visible = grid_visible
        self._show_legend = show_legend
        self._legend_location = legend_location
        self._legend_click_policy = legend_click_policy

        super().__init__(widget_id=widget_id, file_path=__file__)

        self._update()

        server = self._sly_app.get_server()

        @server.get(self.html_route)
        def _html_response() -> None:
            return HTMLResponse(content=self._get_html())

    @property
    def html_route(self) -> str:
        return self.get_route_path(Bokeh.Routes.HTML_ROUTE)

    @property
    def html_route_with_timestamp(self) -> str:
        return f".{self.html_route}?t={datetime.now().timestamp()}"

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def _add_callbacks(self):
        from bokeh.models import CustomJS  # pylint: disable=import-error

        route_path = self.get_route_path(Bokeh.Routes.VALUE_CHANGED)

        if is_production():
            api = Api()
            task_info = api.task.get_info_by_id(env_task_id())
            if task_info is not None:
                route_path = f"/net/{task_info['meta']['sessionToken']}{route_path}"
        callback = CustomJS(
            args=dict(source=self._source),
            code=f"""
                var indices = source.selected.indices;
                var selected_ids = [];
                for (var i = 0; i < indices.length; i++) {{
                    selected_ids.push(source.data['ids'][indices[i]]);
                }}
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "{route_path}", true);
                xhr.setRequestHeader("Content-Type", "application/json");
                xhr.send(JSON.stringify({{selected_ids: selected_ids}}));
            """,
        )
        self._source.selected.js_on_change("indices", callback)

    def _get_html(self) -> str:
        return f"""<div>
            <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.1.1.min.js"></script>
            {self._script}
            {self._div}
        </div>"""

    def _create_figure(self):
        from bokeh.models import (  # pylint: disable=import-error
            ColumnDataSource,
            Legend,
        )
        from bokeh.plotting import figure  # pylint: disable=import-error

        self._plot = figure(width=self._width, height=self._height, tools=self._tools)

        self._plot.xaxis.visible = self._x_axis_visible
        self._plot.yaxis.visible = self._y_axis_visible
        self._plot.grid.visible = self._grid_visible

        self._source = ColumnDataSource(data=self._source_data)
        self._add_callbacks()
        legend_items = []
        for plot in self._plots:
            r = plot.add(self._plot, self._source)
            legend_items.append((plot.name, [r]))

        if self._show_legend:
            self._plot.add_layout(
                Legend(items=legend_items, click_policy=self._legend_click_policy),
                self._legend_location,
            )

    def _update_html(self):
        from bokeh.embed import components  # pylint: disable=import-error

        script, self._div = components(self._plot)
        self._div_id = self._get_div_id(self._div)
        self._script = self._update_script(script)

    def _update(self):
        self._create_figure()
        self._update_html()

    def _get_div_id(self, div: str) -> str:
        match = re.search(r'id="([^"]+)"', div)
        if match:
            return match.group(1)
        raise ValueError(f"No div id found in {div}")

    def _update_script(self, script: str) -> str:
        insert_after = "const fn = function() {"
        updated_script = ""
        for line in script.split("\n"):
            if line.strip().startswith(insert_after):
                line += f"""
                    const el = document.querySelector('#{self._div_id}');
                    if (el === null) {{
                        setTimeout(fn, 200);
                        return;
                    }}
                """
            updated_script += line + "\n"
        return updated_script

    def value_changed(self, func: Callable) -> Callable:
        """Registers a callback function that will be called when the chart is clicked."""

        server = self._sly_app.get_server()
        route_path = self.get_route_path(Bokeh.Routes.VALUE_CHANGED)
        self._changes_handled = True
        debounced_handler = DebouncedEventHandler(debounce_time=0.2)

        @server.post(route_path)
        async def _click(data: SelectedIds) -> None:
            debounced_handler.handle_event(data.selected_ids, func)

        return _click

    def clear(self) -> None:
        """Clears all data in the ColumnDataSource and removes plots."""

        self._source_data = {key: [] for key in self._source_data.keys()}
        self._plots = []
        self._update()

    def refresh(self) -> None:
        """Refreshes the chart by reloading the existing data and updating the HTML."""

        self._update()

    def update_chart_size(self, width: Optional[int] = None, height: Optional[int] = None):
        """Updates the size of the chart."""

        if width:
            self._width = width
        if height:
            self._height = height
        self._update()

    def add_data(
        self,
        x: List[float],
        y: List[float],
        radius: List[float],
        colors: List[str],
        ids: List[Any],
        names: List[str],
        append: bool = True,
    ):
        """Adds data to the chart."""

        if append:
            self._source.data["x"] += x
            self._source.data["y"] += y
            self._source.data["radius"] += radius
            self._source.data["colors"] += colors
            self._source.data["ids"] += ids
            self._source.data["names"] += names
        else:
            self._source.data = {
                "x": x,
                "y": y,
                "radius": radius,
                "colors": colors,
                "ids": ids,
                "names": names,
            }

    def add_plot(self, plot: Plot):
        """Adds a plot to the chart."""

        self._plots.append(plot)
        self._update()

    def add_plots(self, plots: List[Plot]):
        """Adds multiple plots to the chart."""

        self._plots.extend(plots)
        self._update()

    def update_point_size(self, size: float):
        """Updates the size of the points in the chart."""

        self._source_data["radius"] = [size] * len(self._source_data["x"])
        self._update()
