from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    NotificationWidget,
)


class Recall(DetectionVisMetric):
    MARKDOWN = "recall"
    MARKDOWN_PER_CLASS = "recall_per_class"
    NOTIFICATION = "recall"
    CHART = "recall"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_R
        return MarkdownWidget(self.MARKDOWN, "Recall", text)

    @property
    def notification(self) -> NotificationWidget:
        title, desc = self.vis_texts.notification_recall.values()
        tp_plus_fn = self.eval_result.mp.TP_count + self.eval_result.mp.FN_count
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.base_metrics()["recall"].round(2)),
            desc.format(self.eval_result.mp.TP_count, tp_plus_fn),
        )

    @property
    def per_class_md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_R_perclass.format(self.vis_texts.definitions.f1_score)
        return MarkdownWidget(self.MARKDOWN_PER_CLASS, "Recall per class", text)

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

        sorted_by_f1 = self.eval_result.mp.per_class_metrics().sort_values(by="f1")
        fig = px.bar(
            sorted_by_f1,
            x="category",
            y="recall",
            color="recall",
            range_color=[0, 1],
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
        fig.update_layout(
            width=700 if len(sorted_by_f1) < 10 else None,
        )
        return fig
