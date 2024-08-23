from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ClasswiseErrorAnalysis(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.cv_tasks: List[CVTask] = [CVTask.SEMANTIC_SEGMENTATION.value]

    def get_figure(self, widget: Widget.Chart):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        pd.options.mode.chained_assignment = None  # TODO rm later

        df = self._loader.result_df
        df.drop(["mean"], inplace=True)
        df = df[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
        df.sort_values(by="IoU", ascending=False, inplace=True)
        labels = list(df.index)
        color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

        fig = go.Figure()
        for i, column in enumerate(df.columns):
            fig.add_trace(
                go.Bar(
                    name=column,
                    y=df[column],
                    x=labels,
                    marker_color=color_palette[i],
                )
            )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Classwise segmentation error analysis",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
        )
        return fig
