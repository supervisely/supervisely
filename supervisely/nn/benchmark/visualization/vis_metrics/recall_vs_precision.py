from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class RecallVsPrecision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_PR=Widget.Markdown(
                title="Recall vs Precision",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.f1_score],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        blue_color = "#1f77b4"
        orange_color = "#ff7f0e"
        sorted_by_f1 = self._loader.mp.per_class_metrics().sort_values(by="f1")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=sorted_by_f1["precision"],
                x=sorted_by_f1["category"],
                name="Precision",
                marker=dict(color=blue_color),
            )
        )
        fig.add_trace(
            go.Bar(
                y=sorted_by_f1["recall"],
                x=sorted_by_f1["category"],
                name="Recall",
                marker=dict(color=orange_color),
            )
        )
        fig.update_layout(
            barmode="group",
            # title="Per-class Precision and Recall (Sorted by F1)",
        )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        # fig.show()
        return fig
