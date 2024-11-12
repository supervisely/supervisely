from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    NotificationWidget,
)


class Precision(DetectionVisMetric):
    MARKDOWN = "precision"
    MARKDOWN_PER_CLASS = "precision_per_class"
    NOTIFICATION = "precision"
    CHART = "precision"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_P
        return MarkdownWidget(self.MARKDOWN, "Precision", text)

    @property
    def notification(self) -> NotificationWidget:
        title, desc = self.vis_texts.notification_precision.values()
        tp_plus_fp = self.eval_result.mp.TP_count + self.eval_result.mp.FP_count
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.base_metrics()["precision"].round(2)),
            desc.format(self.eval_result.mp.TP_count, tp_plus_fp),
        )

    @property
    def per_class_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_P_perclass.format(self.vis_texts.definitions.f1_score)
        return MarkdownWidget(self.MARKDOWN_PER_CLASS, "Precision per class", text)

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget(self.CHART, self._get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].label}`,",
        )
        return chart

    def _get_figure(self):  #  -> go.Figure
        import plotly.express as px  # pylint: disable=import-error

        sorted_by_precision = self.eval_result.mp.per_class_metrics().sort_values(by="precision")
        fig = px.bar(
            sorted_by_precision,
            x="category",
            y="precision",
            # title="Per-class Precision (Sorted by F1)",
            color="precision",
            range_color=[0, 1],
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
