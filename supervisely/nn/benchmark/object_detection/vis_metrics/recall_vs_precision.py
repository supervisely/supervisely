from __future__ import annotations

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class RecallVsPrecision(BaseVisMetric):
    MARKDOWN = "recall_vs_precision"
    CHART = "recall_vs_precision"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_PR.format(self.vis_texts.definitions.f1_score)
        return MarkdownWidget(self.MARKDOWN, "Recall vs Precision", text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        blue_color = "#1f77b4"
        orange_color = "#ff7f0e"
        sorted_by_f1 = self.eval_result.mp.per_class_metrics().sort_values(by="f1")
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
        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        return fig
