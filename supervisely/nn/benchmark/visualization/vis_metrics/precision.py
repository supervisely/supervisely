from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class Precision(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_P=Widget.Markdown(title="Precision", is_header=True),
            notification_precision=Widget.Notification(
                formats_title=[self._loader.base_metrics()["precision"].round(2)],
                formats_desc=[
                    self._loader.mp.TP_count,
                    (self._loader.mp.TP_count + self._loader.mp.FP_count),
                ],
            ),
            markdown_P_perclass=Widget.Markdown(
                formats=[self._loader.vis_texts.definitions.f1_score]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget):  #  -> Optional[go.Figure]
        import plotly.express as px  # pylint: disable=import-error

        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="precision")
        sorted_by_precision = self._loader.mp.per_class_metrics().sort_values(by="precision")
        fig = px.bar(
            sorted_by_precision,
            x="category",
            y="precision",
            # title="Per-class Precision (Sorted by F1)",
            color="precision",
            color_continuous_scale="Plasma",
        )
        fig.update_traces(hovertemplate="Class: %{x}<br>Precision: %{y:.2f}<extra></extra>")
        if len(sorted_by_precision) <= 20:
            fig.update_traces(
                text=sorted_by_precision.round(2),
                textposition="outside",
            )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Precision", range=[0, 1])
        return fig
