from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
    TableWidget,
)


class CalibrationScore(BaseVisMetric):
    @property
    def header_md(self) -> MarkdownWidget:
        text_template = self.vis_texts.markdown_calibration_score
        text = text_template.format(self.vis_texts.definitions.confidence_score)
        return MarkdownWidget(
            name="markdown_calibration_score",
            title="Calibration Score",
            text=text,
        )

    @property
    def collapse_tip(self) -> CollapseWidget:
        md = MarkdownWidget(
            name="what_is_calibration",
            title="What is calibration?",
            text=self.vis_texts.markdown_what_is_calibration,
        )
        return CollapseWidget([md])

    @property
    def header_md_2(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="markdown_calibration_score_2",
            title="",
            text=self.vis_texts.markdown_calibration_score_2,
        )

    @property
    def table(self) -> TableWidget:
        TableWidget()

    @property
    def reliability_diagram_md(self) -> MarkdownWidget:
        return MarkdownWidget(
            name="markdown_reliability_diagram",
            title="Reliability Diagram",
            text=self.vis_texts.markdown_reliability_diagram,
        )

    @property
    def notification_ece(self) -> NotificationWidget:
        desc = "\n".join(
            f"{ev.name}: {ev.mp.m_full.expected_calibration_error():.4f}"
            for ev in self.eval_results
        )
        return NotificationWidget(
            name="notification_ece",
            title="Expected Calibration Error (ECE):",
            desc=desc,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name="chart", figure=self.get_figure(), click_data=None)

    @property
    def collapse_ece(self) -> CollapseWidget:
        md = MarkdownWidget(
            name="markdown_calibration_curve_interpretation",
            title="How to interpret the Calibration curve",
            text=self.vis_texts.markdown_calibration_curve_interpretation,
        )
        return CollapseWidget([md])

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        for eval_result in self.eval_results:
            # Calibration curve (only positive predictions)
            true_probs, pred_probs = eval_result.mp.m_full.calibration_curve()

            fig.add_trace(
                go.Scatter(
                    x=pred_probs,
                    y=true_probs,
                    mode="lines+markers",
                    name="Calibration plot (Model)",
                    line=dict(color="blue"),
                    marker=dict(color="blue"),
                    hovertemplate=f"{eval_result.name}<br>"
                    + "Confidence Score: %{x:.2f}<br>Fraction of True Positives: %{y:.2f}<extra></extra>",
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
            # title="Calibration Curve (only positive predictions)",
            xaxis_title="Confidence Score",
            yaxis_title="Fraction of True Positives",
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=500,
        )
        return fig
