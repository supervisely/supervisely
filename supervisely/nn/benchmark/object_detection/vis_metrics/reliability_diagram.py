from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
)


class ReliabilityDiagram(DetectionVisMetric):
    MARKDOWN_CALIBRATION_SCORE = "calibration_score"
    MARKDOWN_CALIBRATION_SCORE_2 = "calibration_score_2"
    MARKDOWN_RELIABILITY_DIAGRAM = "reliability_diagram"
    NOTIFICATION = "reliability_diagram"
    CHART = "reliability_diagram"

    @property
    def md_calibration_score(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_calibration_score_1
        text = text.format(self.vis_texts.definitions.confidence_score)
        return MarkdownWidget(self.MARKDOWN_CALIBRATION_SCORE, "Calibration Score", text)

    @property
    def collapse_tip(self) -> CollapseWidget:
        md = MarkdownWidget(
            "what_is_calibration",
            "What is calibration?",
            self.vis_texts.markdown_what_is_calibration,
        )
        return CollapseWidget([md])

    @property
    def md_calibration_score_2(self) -> MarkdownWidget:
        return MarkdownWidget(
            self.MARKDOWN_CALIBRATION_SCORE_2,
            "",
            self.vis_texts.markdown_calibration_score_2,
        )

    @property
    def md_reliability_diagram(self) -> MarkdownWidget:
        return MarkdownWidget(
            self.MARKDOWN_RELIABILITY_DIAGRAM,
            "Reliability Diagram",
            self.vis_texts.markdown_reliability_diagram,
        )

    @property
    def notification(self) -> NotificationWidget:
        title, _ = self.vis_texts.notification_ece.values()
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.m_full.expected_calibration_error().round(4)),
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    @property
    def collapse(self) -> CollapseWidget:
        md = MarkdownWidget(
            "markdown_calibration_curve_interpretation",
            "How to interpret the Calibration curve",
            self.vis_texts.markdown_calibration_curve_interpretation,
        )
        return CollapseWidget([md])

    def _get_figure(self):  # -> go.Figure:
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Calibration curve (only positive predictions)
        true_probs, pred_probs = self.eval_result.mp.m_full.calibration_curve()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pred_probs,
                y=true_probs,
                mode="lines+markers",
                name="Calibration plot (Model)",
                line=dict(color="blue"),
                marker=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfectly calibrated",
                line=dict(color="orange", dash="dash"),
            )
        )

        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Fraction of True Positives",
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=500,
        )
        fig.update_traces(
            hovertemplate="Confidence Score: %{x:.2f}<br>Fraction of True Positives: %{y:.2f}<extra></extra>"
        )
        return fig
