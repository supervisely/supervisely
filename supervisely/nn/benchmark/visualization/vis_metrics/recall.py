from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class Recall(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        tp_plus_fn = self._loader.mp.TP_count + self._loader.mp.FN_count
        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_R=Widget.Markdown(title="Recall", is_header=True),
            notification_recall=Widget.Notification(
                formats_title=[self._loader.base_metrics()["recall"].round(2)],
                formats_desc=[self._loader.mp.TP_count, tp_plus_fn],
            ),
            markdown_R_perclass=Widget.Markdown(
                formats=[self._loader.vis_texts.definitions.f1_score]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]
        import plotly.express as px  # pylint: disable=import-error

        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="recall")
        sorted_by_f1 = self._loader.mp.per_class_metrics().sort_values(by="f1")
        fig = px.bar(
            sorted_by_f1,
            x="category",
            y="recall",
            # title="Per-class Recall (Sorted by F1)",
            color="recall",
            color_continuous_scale="Plasma",
        )
        fig.update_traces(hovertemplate="Class: %{x}<br>Recall: %{y:.2f}<extra></extra>")
        if len(sorted_by_f1) <= 20:
            fig.update_traces(
                text=sorted_by_f1["recall"].round(2),
                textposition="outside",
            )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Recall", range=[0, 1])
        return fig
