from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class PerClassAvgPrecision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_class_ap=Widget.Markdown(
                title="Average Precision by Class",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.average_precision],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        # AP per-class
        ap_per_class = self._loader.mp.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
        ap_per_class[ap_per_class == -1] = 0  # -1 is a placeholder for no GT
        labels = dict(r="Average Precision", theta="Class")
        fig = px.scatter_polar(
            r=ap_per_class,
            theta=self._loader.mp.cat_names,
            # title="Per-class Average Precision (AP)",
            labels=labels,
            width=800,
            height=800,
            range_r=[0, 1],
        )
        fig.update_traces(fill="toself")
        fig.update_layout(
            modebar_add=["resetScale"],
            margin=dict(l=80, r=80, t=0, b=0),
        )
        fig.update_traces(
            hovertemplate=labels["theta"]
            + ": %{theta}<br>"
            + labels["r"]
            + ": %{r:.2f}<br>"
            + "<extra></extra>"
        )
        return fig
