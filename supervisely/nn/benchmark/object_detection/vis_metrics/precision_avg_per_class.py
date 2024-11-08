from __future__ import annotations

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class PerClassAvgPrecision(BaseVisMetric):
    MARKDOWN = "per_class_avg_precision"
    CHART = "per_class_avg_precision"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_class_ap
        text = text.format(self.vis_texts.definitions.average_precision)
        return MarkdownWidget(self.MARKDOWN, "Average Precision by Class", text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        # AP per-class
        ap_per_class = self.eval_result.mp.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
        ap_per_class[ap_per_class == -1] = 0  # -1 is a placeholder for no GT
        labels = dict(r="Average Precision", theta="Class")
        fig = px.scatter_polar(
            r=ap_per_class,
            theta=self.eval_result.mp.cat_names,
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
