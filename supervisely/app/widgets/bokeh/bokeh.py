from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from uuid import uuid4

from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.io.env import task_id
from supervisely.sly_logger import logger


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
    plot_id: Optional[Union[str, int]] = None


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

        plot = Bokeh.Circle(
            x_coordinates=[1, 2, 3, 4, 5],
            y_coordinates=[1, 2, 3, 4, 5],
            radii=10,
            colors="red",
            legend_label="Circle plot",
        )
        bokeh = Bokeh(plots=[plot], width=1000, height=600)

        # To allow the widget to be interacted with, you need to add it to the IFrame widget.
        iframe = IFrame()
        iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
    """

    class Routes:
        VALUE_CHANGED = "value_changed"
        HTML_ROUTE = "bokeh.html"

    class Plot(ABC):
        @abstractmethod
        def add(self, plot) -> None:
            pass

        @abstractmethod
        def register(self, route_path: str) -> None:
            pass

    class Circle(Plot):
        def __init__(
            self,
            x_coordinates: List[Union[int, float]],
            y_coordinates: List[Union[int, float]],
            radii: Optional[Union[Union[int, float], List[Union[int, float]]]] = None,
            colors: Optional[Union[str, List[str]]] = None,
            data: Optional[List[Any]] = None,
            dynamic_selection: bool = False,
            fill_alpha: float = 0.5,
            line_color: Optional[str] = None,
            legend_label: Optional[str] = None,
            plot_id: Optional[Union[str, int]] = None,
        ):
            if not colors:
                colors = Bokeh._generate_colors(x_coordinates, y_coordinates)
            elif isinstance(colors, str):
                colors = [colors] * len(x_coordinates)

            if not radii:
                radii = Bokeh._generate_radii(x_coordinates, y_coordinates)
            elif isinstance(radii, (int, float)):
                radii = [radii] * len(x_coordinates)

            if not len(x_coordinates) == len(y_coordinates) == len(radii) == len(colors):
                raise ValueError(
                    "x_coordinates, y_coordinates, radii, and colors must have the same length"
                )

            if data is not None and len(data) != len(x_coordinates):
                raise ValueError("data must have the same length as x_coordinates")

            if data is None:
                data = list(range(len(x_coordinates)))

            self._x_coordinates = x_coordinates
            self._y_coordinates = y_coordinates
            self._radii = radii
            self._colors = colors
            self._data = data
            self._source = None
            self._dynamic_selection = dynamic_selection
            self._fill_alpha = fill_alpha
            self._line_color = line_color
            self._plot_id = plot_id or uuid4().hex
            self._legend_label = legend_label or str(self._plot_id)

        def add(self, plot) -> None:
            from bokeh.models import (  # pylint: disable=import-error
                ColumnDataSource,
                LassoSelectTool,
            )

            data = dict(
                x=self._x_coordinates,
                y=self._y_coordinates,
                radius=self._radii,
                colors=self._colors,
                ids=self._data,
            )
            self._source = ColumnDataSource(data=data)

            renderer = plot.circle(
                "x",
                "y",
                radius="radius",
                fill_color="colors",
                fill_alpha=self._fill_alpha,
                line_color=self._line_color,
                source=self._source,
                legend_label=self._legend_label,
            )
            if not self._dynamic_selection:
                for tool in plot.tools:
                    if isinstance(tool, (LassoSelectTool)):
                        tool.continuous = False

            return renderer

        def register(self, route_path: str) -> None:
            from bokeh.models import CustomJS  # pylint: disable=import-error

            if not hasattr(self, "_source"):
                raise ValueError("Plot must be added to a Bokeh plot before registering")

            if is_production():
                api = Api()
                task_info = api.task.get_info_by_id(task_id())
                if task_info is not None:
                    route_path = f"/net/{task_info['meta']['sessionToken']}{route_path}"
            callback = CustomJS(
                args=dict(source=self._source),
                code="""
                    var indices = source.selected.indices;
                    var selected_ids = [];
                    for (var i = 0; i < indices.length; i++) {{
                        selected_ids.push(source.data['ids'][indices[i]]);
                    }}
                    var xhr = new XMLHttpRequest();
                    xhr.open("POST", "{route_path}", true);
                    xhr.setRequestHeader("Content-Type", "application/json");
                    xhr.send(JSON.stringify({{selected_ids: selected_ids, plot_id: '{plot_id}'}}));
                """.format(
                    route_path=route_path,
                    plot_id=self._plot_id,
                ),
            )
            self._source.selected.js_on_change("indices", callback)

    def __init__(
        self,
        plots: List[Plot],
        width: int = 1000,
        height: int = 600,
        tools: List[str] = [
            "pan",
            "wheel_zoom",
            "box_zoom",
            "reset",
            "save",
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

        self.widget_id = widget_id
        self._plots = plots

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
        self._load_chart()

        server = self._sly_app.get_server()

        @server.get(self.html_route)
        def _html_response() -> None:
            return HTMLResponse(content=self.get_html())

        # TODO: support for offline mode
        # JinjaWidgets().context.pop(self.widget_id, None)  # remove the widget from index.html

    def _load_chart(self, **kwargs):
        from bokeh.models import Legend  # pylint: disable=import-error
        from bokeh.plotting import figure  # pylint: disable=import-error

        self._width = kwargs.get("width", self._width)
        self._height = kwargs.get("height", self._height)
        self._tools = kwargs.get("tools", self._tools)
        self._toolbar_location = kwargs.get("toolbar_location", self._toolbar_location)
        self._show_legend = kwargs.get("show_legend", self._show_legend)
        self._legend_location = kwargs.get("legend_location", self._legend_location)
        self._legend_click_policy = kwargs.get("legend_click_policy", self._legend_click_policy)
        self._x_axis_visible = kwargs.get("x_axis_visible", self._x_axis_visible)
        self._y_axis_visible = kwargs.get("y_axis_visible", self._y_axis_visible)
        self._grid_visible = kwargs.get("grid_visible", self._grid_visible)

        self._plot = figure(
            width=self._width,
            height=self._height,
            tools=self._tools,
            toolbar_location=self._toolbar_location,
        )

        if self._show_legend:
            self._plot.add_layout(
                Legend(click_policy=self._legend_click_policy),
                self._legend_location,
            )

        self._plot.xaxis.visible = self._x_axis_visible
        self._plot.yaxis.visible = self._y_axis_visible
        self._plot.grid.visible = self._grid_visible

        self._renderers = []
        self._process_plots(self._plots)
        self._update_html()

    @property
    def route_path(self) -> str:
        return self.get_route_path(Bokeh.Routes.VALUE_CHANGED)

    @property
    def html_route(self) -> str:
        return self.get_route_path(Bokeh.Routes.HTML_ROUTE)

    @property
    def html_route_with_timestamp(self) -> str:
        return f".{self.html_route}?t={datetime.now().timestamp()}"

    def add_plots(self, plots: List[Plot]) -> None:
        self._plots.extend(plots)
        self._process_plots(plots)
        self._update_html()

    def clear(self) -> None:
        self._plots = []
        self._renderers = []
        self._plot.renderers = []
        self._update_html()

    def remove_plot(self, idx: int) -> None:
        renderer = self._renderers.pop(idx)
        self._plot.renderers.remove(renderer)
        self._update_html()

    def _process_plots(self, plots: List[Plot]) -> None:
        for plot in plots:
            renderer = plot.add(self._plot)
            plot.register(self.route_path)
            self._renderers.append(renderer)

    def _update_html(self) -> None:
        from bokeh.embed import components  # pylint: disable=import-error

        script, self._div = components(self._plot, wrap_script=False)
        self._div_id = self._get_div_id(self._div)
        self._script = self._update_script(script)

    @staticmethod
    def _generate_colors(x_coordinates: List[int], y_coordinates: List[int]) -> List[str]:
        colors = [
            "#%02x%02x%02x" % (int(r), int(g), 150)
            for r, g in zip(
                [50 + 2 * xi for xi in x_coordinates], [30 + 2 * yi for yi in y_coordinates]
            )
        ]
        return colors

    @staticmethod
    def _generate_radii(x_coordinates: List[int], y_coordinates: List[int]) -> List[int]:
        return [1] * len(x_coordinates)

    def _get_div_id(self, div: str) -> str:
        match = re.search(r'id="([^"]+)"', div)
        if match:
            return match.group(1)
        raise ValueError(f"No div id found in {div}")

    def _update_script(self, script: str) -> str:
        # TODO: Reimplement using regex.
        insert_after = "const fn = function() {"
        updated_script = ""
        for line in script.split("\n"):
            if line.strip().startswith(insert_after):
                line = line + f"\n    const el = document.querySelector('#{self._div_id}');"
                line += "\n    if (el === null) {"
                line += "\n      setTimeout(fn, 500);"
                line += "\n      return;"
                line += "\n    }"
            updated_script += line + "\n"
        return updated_script

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def value_changed(self, func: Callable) -> Callable:
        server = self._sly_app.get_server()
        self._changes_handled = True
        debounced_handler = DebouncedEventHandler(debounce_time=0.2)  # TODO: check if it's enough

        @server.post(self.route_path)
        async def _click(data: SelectedIds) -> None:
            debounced_handler.handle_event(data, func)

        return _click

    def get_html(self) -> str:
        return f"""<div>
            <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.1.1.min.js"></script>
            <script type="text/javascript"> {self._script} </script>
            {self._div}
        </div>"""

    def update_radii(self, new_radii: Union[List[Union[list, int, float]], int, float]) -> None:
        if isinstance(new_radii, (int, float)):
            new_radii = [new_radii] * len(self._plots)
        elif len(new_radii) != len(self._plots):
            logger.warning(
                f"{len(new_radii)} != {len(self._plots)}: new_radii will be broadcasted to all plots"
            )
            new_radii = [new_radii[0]] * len(self._plots)
        for idx, radii in enumerate(new_radii):
            self.update_radii_by_plot_idx(idx, radii)

    def update_radii_by_plot_idx(self, plot_idx: int, new_radii: List[Union[int, float]]) -> None:
        coords_length = len(self._plots[plot_idx]._x_coordinates)
        if isinstance(new_radii, (int, float)):
            new_radii = [new_radii] * coords_length
        elif len(new_radii) != coords_length:
            logger.warning(
                f"{len(new_radii)} != {coords_length}: new_radii will be broadcasted to all plots"
            )
            new_radii = [new_radii[0]] * coords_length

        self._plots[plot_idx]._radii = new_radii
        self._plots[plot_idx]._source.data["radius"] = new_radii
        self._load_chart()

    def update_chart_size(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self._width = width or self._width
        self._height = height or self._height
        self._load_chart()
