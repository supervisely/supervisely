from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
)


class ConfidenceScore(DetectionVisMetric):
    MARKDOWN_CONFIDENCE_SCORE = "confidence_score"
    NOTIFICATION = "confidence_score"
    MARKDOWN_CONFIDENCE_SCORE_2 = "confidence_score_2"
    MARKDOWN_CONFIDENCE_SCORE_3 = "calibration_score_3"
    COLLAPSE_TITLE = "confidence_score_collapse"
    CHART = "confidence_score"

    @property
    def md_confidence_score(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_confidence_score_1
        text = text.format(self.vis_texts.definitions.confidence_threshold)
        return MarkdownWidget(self.MARKDOWN_CONFIDENCE_SCORE, "Confidence Score Profile", text)

    @property
    def notification(self) -> NotificationWidget:
        title, _ = self.vis_texts.notification_f1.values()
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.f1_optimal_conf.round(4)),
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name=self.CHART, figure=self._get_figure())

    @property
    def md_confidence_score_2(self) -> MarkdownWidget:
        return MarkdownWidget(
            self.MARKDOWN_CONFIDENCE_SCORE_2,
            "",
            self.vis_texts.markdown_confidence_score_2,
        )

    @property
    def collapse_conf_score(self) -> CollapseWidget:
        md = MarkdownWidget(
            self.COLLAPSE_TITLE,
            "How to plot Confidence Profile?",
            self.vis_texts.markdown_plot_confidence_profile,
        )
        return CollapseWidget([md])

    @property
    def md_confidence_score_3(self) -> MarkdownWidget:
        return MarkdownWidget(
            self.MARKDOWN_CONFIDENCE_SCORE_3,
            "",
            self.vis_texts.markdown_calibration_score_3,
        )

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        color_map = {
            "Precision": "#1f77b4",
            "Recall": "orange",
        }

        fig = px.line(
            self.eval_result.dfsp_down,
            x="scores",
            y=["precision", "recall", "f1"],
            labels={"value": "Metric", "variable": "Metric", "scores": "Confidence Score"},
            width=None,
            height=500,
            color_discrete_map=color_map,
        )
        fig.update_traces(
            hovertemplate="Confidence score: %{x:.2f}<br>Metric: %{y:.2f}<extra></extra>"
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        if (
            self.eval_result.mp.f1_optimal_conf is not None
            and self.eval_result.mp.best_f1 is not None
        ):
            # Add vertical line for the best threshold
            fig.add_shape(
                type="line",
                x0=self.eval_result.mp.f1_optimal_conf,
                x1=self.eval_result.mp.f1_optimal_conf,
                y0=0,
                y1=self.eval_result.mp.best_f1,
                line=dict(color="gray", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=self.eval_result.mp.f1_optimal_conf,
                y=self.eval_result.mp.best_f1 + 0.04,
                text=f"F1-optimal threshold: {self.eval_result.mp.f1_optimal_conf:.2f}",
                showarrow=False,
            )
        if self.eval_result.mp.custom_conf_threshold is not None:
            # Add vertical line for the custom threshold
            fig.add_shape(
                type="line",
                x0=self.eval_result.mp.custom_conf_threshold,
                x1=self.eval_result.mp.custom_conf_threshold,
                y0=0,
                y1=self.eval_result.mp.custom_f1,
                line=dict(color="black", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=self.eval_result.mp.custom_conf_threshold,
                y=self.eval_result.mp.custom_f1 + 0.04,
                text=f"Confidence threshold: {self.eval_result.mp.custom_conf_threshold:.2f}",
                showarrow=False,
            )
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
