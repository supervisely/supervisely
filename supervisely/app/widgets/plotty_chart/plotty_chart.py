from __future__ import annotations

from typing import Any, List, Literal, NamedTuple, Optional, Union
from supervisely.app.widgets import Widget
from supervisely.app.content import StateJson, DataJson
import json


class PlottyChart(Widget):
    class Routes:
        CLICK = "chart_clicked_cb"

    class ClickedDataPoint(NamedTuple):
        """
        Class, representing clicked datapoint, which contains information about chart points.
        It will be returned after click event on datapoint in immutable namedtuple
        """

        type: Optional[str]
        x: Optional[int]
        y: Optional[int]
        z: Optional[int]
        i: Optional[int]
        curveNumber: Optional[int]
        pointIndex: Optional[int]
        label: Optional[str]
        color: Optional[str]
        percent: Optional[float]
        pointNumber: Optional[int]
        pointNumbers: Optional[List[int]]
        value: Optional[Union[int, float, str]]
        v: Optional[Union[int, float, str]]
        customdata: Optional[Any]
        marker_size: Optional[float]
        marker_color: Optional[float]
        # data: Dict[str, Any] # too big to store in state?

    @staticmethod
    def datapoint_sequence():
        return [
            "type",
            "x",
            "y",
            "z",
            "i",
            "curveNumber",
            "pointIndex",
            "label",
            "color",
            "percent",
            "pointNumber",
            "pointNumbers",
            "value",
            "v",
            "customdata",
            "marker_size",
            "marker_color",
            # "data"
        ]

    def __init__(
        self,
        figure=None,  # Union[plotly.graph_objects.Figure, dict, None],
        element_loading_text: str = "Loading...",
        widget_id: str = None,
    ):
        self._figure = self._validate_figure(figure if figure is not None else {})
        self._click_handled = False
        self._element_loading_text = element_loading_text
        self._loading = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _validate_figure(self, figure):
        import plotly.graph_objects as go  # pylint: disable=import-error
        from plotly.io import from_json  # pylint: disable=import-error
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

        if isinstance(figure, go.Figure):
            figure = figure.to_dict()
        elif isinstance(figure, dict):
            # try to convert to Figure object to check if it is valid
            fig = from_json(json.dumps(figure), skip_invalid=True)
            figure = fig.to_dict()
        else:
            raise ValueError(f"figure must be Plotly Figure or dict, but {type(figure)} was given.")

        return _convert_arrays_to_lists(figure)

    def get_json_data(self):
        return {
            "figure": self._figure,
            "options": {"elementLoadingText": self._element_loading_text},
        }

    def get_json_state(self):
        return {
            "clicked_value": None,
            "loading": self._loading,
        }

    def click(self, func):
        route_path = self.get_route_path(PlottyChart.Routes.CLICK)
        server = self._sly_app.get_server()

        self._click_handled = True

        @server.post(route_path)
        def _click():
            res = self.get_clicked_datapoint()
            if res is not None:
                return func(res)

        return _click

    def get_clicked_datapoint(self) -> Union[PlottyChart.ClickedDataPoint, None]:
        points = StateJson()[self.widget_id]["clicked_value"]
        if not isinstance(points, list):
            return None
        res = []
        for point_data in points:
            data = point_data.pop("data", None)
            point_data["type"] = data.get("type") if data is not None else None
            point_data["label"] = data.get("name") if data is not None else None
            fields = []
            for field in self.datapoint_sequence():
                if field.replace("_", ".") in point_data:
                    fields.append(point_data.get(field.replace("_", ".")))
                else:
                    fields.append(point_data.get(field))
            res.append(PlottyChart.ClickedDataPoint(*fields))
        return res

    def set_figure(self, figure):
        self._figure = self._validate_figure(figure)
        DataJson()[self.widget_id]["figure"] = self._figure
        DataJson().send_changes()

    def get_figure(self, return_type: Literal["dict", "plottly_figure"] = "plottly_figure"):
        if return_type == "dict":
            return self._figure
        elif return_type == "plottly_figure":
            from plotly.io import from_json  # pylint: disable=import-error
            from plotly.utils import PlotlyJSONEncoder  # pylint: disable=import-error

            figure = DataJson()[self.widget_id]["figure"]
            figure = json.dumps(figure, cls=PlotlyJSONEncoder)
            return from_json(figure, skip_invalid=True)
        else:
            raise ValueError(
                f"return_type must be 'dict' or 'figure', but {return_type} was given."
            )
