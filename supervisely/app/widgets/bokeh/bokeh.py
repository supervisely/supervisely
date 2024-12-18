from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supervisely.app.widgets import Widget
from supervisely._utils import is_production
from supervisely.io.env import task_id
from supervisely.api.api import Api


class SelectedIds(BaseModel):
    selected_ids: List[Any]


class Bokeh(Widget):
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
                    xhr.send(JSON.stringify({{selected_ids: selected_ids}}));
                """.format(
                    route_path=route_path
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
        **kwargs,
    ):
        from bokeh.plotting import figure  # pylint: disable=import-error

        self.widget_id = widget_id
        self._plots = plots
        self._plot = figure(width=width, height=height, tools=tools, toolbar_location="above")
        self._renderers = []

        self._plot.xaxis.visible = x_axis_visible
        self._plot.yaxis.visible = y_axis_visible
        self._plot.grid.visible = grid_visible
        super().__init__(widget_id=widget_id, file_path=__file__)

        self._process_plots(plots)
        self._update_html()

        server = self._sly_app.get_server()

        @server.get(self.html_route)
        def _html_response() -> None:
            return HTMLResponse(content=self.get_html())

        # JinjaWidgets().context.pop(self.widget_id, None)  # remove the widget from index.html

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

        @server.post(self.route_path)
        def _click(selected_ids: SelectedIds) -> None:
            func(selected_ids.selected_ids)

        return _click

    def get_html(self) -> str:
        return f"""<div>
            <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.1.1.min.js"></script>
            <script type="text/javascript"> {self._script} </script>
            {self._div}
        </div>"""
