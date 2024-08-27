from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ReliabilityDiagram(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_calibration_score_1=Widget.Markdown(
                title="Calibration Score",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.confidence_score],
            ),
            collapse_what_is=Widget.Collapse(
                Schema(
                    self._loader.vis_texts,
                    markdown_what_is_calibration=Widget.Markdown(title="What is calibration?"),
                )
            ),
            markdown_calibration_score_2=Widget.Markdown(),
            markdown_reliability_diagram=Widget.Markdown(
                title="Reliability Diagram", is_header=True
            ),
            notification_ece=Widget.Notification(
                formats_title=[self._loader.mp.m_full.expected_calibration_error().round(4)]
            ),
            chart=Widget.Chart(),
            collapse_ece=Widget.Collapse(
                Schema(
                    self._loader.vis_texts,
                    markdown_calibration_curve_interpretation=Widget.Markdown(
                        title="How to interpret the Calibration curve"
                    ),
                )
            ),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Calibration curve (only positive predictions)
        true_probs, pred_probs = self._loader.mp.m_full.calibration_curve()

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
            # title="Calibration Curve (only positive predictions)",
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
        # fig.show()
        return fig
