from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ConfidenceDistribution(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_confidence_distribution=Widget.Markdown(
                title="Confidence Distribution",
                is_header=True,
                formats=[
                    self._loader.vis_texts.definitions.true_positives,
                    self._loader.vis_texts.definitions.false_positives,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        f1_optimal_conf, best_f1 = self._loader.mp.m_full.get_f1_optimal_conf()

        # Histogram of confidence scores (TP vs FP)
        scores_tp, scores_fp = self._loader.mp.m_full.scores_tp_and_fp()

        tp_y, tp_x = np.histogram(scores_tp, bins=40, range=[0, 1])
        fp_y, fp_x = np.histogram(scores_fp, bins=40, range=[0, 1])
        dx = (tp_x[1] - tp_x[0]) / 2

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=scores_fp,
                name="FP",
                marker=dict(color="#dd3f3f"),
                opacity=0.5,
                xbins=dict(size=0.025, start=0.0, end=1.0),
                hovertemplate="Confidence Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=scores_tp,
                name="TP",
                marker=dict(color="#1fb466"),
                opacity=0.5,
                xbins=dict(size=0.025, start=0.0, end=1.0),
                hovertemplate="Confidence Score: %{x:.2f}<br>Count: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tp_x + dx,
                y=tp_y,
                mode="lines+markers",
                name="TP",
                line=dict(color="#1fb466", width=2),
                hovertemplate="Confidence Score: %{x:.2f}<br>Count: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fp_x + dx,
                y=fp_y,
                mode="lines+markers",
                name="FP",
                line=dict(color="#dd3f3f", width=2),
                hovertemplate="Confidence Score: %{x:.2f}<br>Count: %{y:.2f}<extra></extra>",
            )
        )

        if f1_optimal_conf is not None:

            # Best threshold
            fig.add_shape(
                type="line",
                x0=f1_optimal_conf,
                x1=f1_optimal_conf,
                y0=0,
                y1=tp_y.max() * 1.3,
                line=dict(color="orange", width=1, dash="dash"),
            )
            fig.add_annotation(
                x=f1_optimal_conf,
                y=tp_y.max() * 1.3,
                text=f"F1-optimal threshold: {f1_optimal_conf:.2f}",
                showarrow=False,
            )

            fig.update_layout(
                barmode="overlay",
                # title="Histogram of Confidence Scores (TP vs FP)",
                width=800,
                height=500,
            )
            fig.update_xaxes(title_text="Confidence Score", range=[0, 1])
            fig.update_yaxes(title_text="Count", range=[0, tp_y.max() * 1.3])
        return fig
