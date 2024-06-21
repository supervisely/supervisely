from __future__ import annotations

from typing import Dict,  List, Literal, NamedTuple, Union
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson
from supervisely.sly_logger import logger
import json


class PlottyChart(Widget):
    class Routes:
        CLICK = "chart_clicked_cb"

    class ClickedDataPoint(NamedTuple):
        """Class, representing clicked datapoint, which contains information about series, data index and data itself.
        It will be returned after click event on datapoint in immutable namedtuple
        with fields: x, y, value."""

        x: int
        y: int
        value: Union[int, float, str]


    def __init__(
        self,
        content, # Union[go.Figure, px.Figure, dict],
        element_loading_text: str = "Loading...",
        widget_id: str = None,
    ):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.io import from_json
        from plotly.utils import PlotlyJSONEncoder
        import numpy as np

        def _convert_arrays_to_lists(d):
            for key, value in d.items():
                if isinstance(value, np.ndarray):
                    d[key] = value.tolist()
                elif isinstance(value, dict):
                    _convert_arrays_to_lists(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            value[i] = item.tolist()
                        elif isinstance(item, dict):
                            _convert_arrays_to_lists(item)
            return d

        if isinstance(content, go.Figure):
            content = content.to_dict()
        elif isinstance(content, dict):
            # try to convert to Figure object to check if it is valid
            fig = from_json(json.dumps(content, PlotlyJSONEncoder), skip_invalid=True)
            content = fig.to_dict()

        self._content = _convert_arrays_to_lists(content)

        self._click_handled = False
        self._element_loading_text = element_loading_text
        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "content": self._content,
            "options": {
                "elementLoadingText": self._element_loading_text
            }
        }

    def get_json_state(self):
        return {
            "clicked_value": None,
            "loading": self._loading,
        }

    def get_clicked_value(self):
        return StateJson()[self.widget_id]["clicked_value"]

    def click(self, func):
        route_path = self.get_route_path(PlottyChart.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            # res = self.get_clicked_datapoint()
            res = self.get_clicked_value()
            if res is not None:
                return func(res)

        return _click

    # def get_clicked_datapoint(self) -> Union[PlottyChart.ClickedDataPoint, None]:
    #     value = self.get_clicked_value()
    #     res = PlottyChart.ClickedDataPoint(
    #         value["x"], value["y"], value["value"]
    #     )
    #     return res

    def set_content(self, content):
        self._content = content
        DataJson()[self.widget_id]["content"] = self._content
        DataJson().send_changes()

    def get_content(self, return_type: Literal["dict", "figure"] = "figure"):
        if return_type == "dict":
            return self._content
        elif return_type == "figure":
            from plotly.io import from_json
            from plotly.utils import PlotlyJSONEncoder
            return from_json(json.dumps(self._content, PlotlyJSONEncoder), skip_invalid=True)
        else:
            raise ValueError(f"return_type must be 'dict' or 'figure', but {return_type} was given.")
