from __future__ import annotations

import pandas as pd

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class PRCurveByClass(BaseVisMetric):
    MARKDOWN = "pr_curve_by_class"
    CHART = "pr_curve_by_class"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_pr_by_class
        return MarkdownWidget(self.MARKDOWN, "PR Curve by Class", text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        df = pd.DataFrame(self.eval_result.mp.pr_curve(), columns=self.eval_result.mp.cat_names)

        fig = px.line(
            df,
            x=self.eval_result.mp.recThrs,
            y=df.columns,
            labels={"x": "Recall", "value": "Precision", "variable": "Category"},
            color_discrete_sequence=px.colors.qualitative.Prism,
            width=800,
            height=600,
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(range=[0, 1])
        return fig
