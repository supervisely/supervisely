from __future__ import annotations

import numpy as np

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
)


class PRCurve(DetectionVisMetric):
    MARKDOWN = "pr_curve"
    NOTIFICATION = "pr_curve"
    COLLAPSE = "pr_curve"
    CHART = "pr_curve"

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_pr_curve.format(self.vis_texts.definitions.about_pr_tradeoffs)
        return MarkdownWidget(self.MARKDOWN, "Precision-Recall Curve", text)

    @property
    def notification(self) -> NotificationWidget:
        title, _ = self.vis_texts.notification_ap.values()
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.base_metrics()["mAP"].round(2)),
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    @property
    def collapse(self) -> CollapseWidget:
        md1 = MarkdownWidget(
            "pr_curve",
            "About Trade-offs between precision and recall",
            self.vis_texts.markdown_trade_offs,
        )
        md2 = MarkdownWidget(
            "what_is_pr_curve",
            "How the PR curve is built?",
            self.vis_texts.markdown_what_is_pr_curve.format(
                self.vis_texts.definitions.confidence_score,
                self.vis_texts.definitions.true_positives,
                self.vis_texts.definitions.false_positives,
            ),
        )
        return CollapseWidget([md1, md2])

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error
        import plotly.graph_objects as go  # pylint: disable=import-error

        pr_curve = self.eval_result.mp.pr_curve().copy()
        pr_curve[pr_curve == -1] = np.nan  # -1 is a placeholder for no GT
        pr_curve = np.nanmean(pr_curve, axis=-1)
        fig = px.line(
            x=self.eval_result.mp.recThrs,
            y=pr_curve,
            labels={"x": "Recall", "y": "Precision"},
            width=600,
            height=500,
        )
        fig.data[0].name = "Model"
        fig.data[0].showlegend = True
        fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))
        fig.add_trace(
            go.Scatter(
                x=self.eval_result.mp.recThrs,
                y=[1] * len(self.eval_result.mp.recThrs),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
        fig.add_annotation(
            text=f"mAP = {self.eval_result.mp.base_metrics()['mAP']:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.92,
            showarrow=False,
            bgcolor="white",
        )
        fig.update_traces(hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>")
        fig.update_layout(
            dragmode=False,
            modebar=dict(
                remove=[
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )
        return fig
