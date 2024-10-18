from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ConfidenceScore(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_confidence_score_1=Widget.Markdown(
                title="Confidence Score Profile",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.confidence_threshold],
            ),
            notification_f1=Widget.Notification(
                formats_title=[round((self._loader.mp.m_full.get_f1_optimal_conf()[0] or 0.0), 4)]
            ),
            chart=Widget.Chart(),
            markdown_confidence_score_2=Widget.Markdown(),
            collapse_conf_score=Widget.Collapse(
                Schema(
                    self._loader.vis_texts,
                    markdown_plot_confidence_profile=Widget.Markdown(
                        title="How to plot Confidence Profile?"
                    ),
                )
            ),
            markdown_calibration_score_3=Widget.Markdown(),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        color_map = {
            "Precision": "#1f77b4",
            "Recall": "orange",
        }

        fig = px.line(
            self._loader.dfsp_down,
            x="scores",
            y=["precision", "recall", "f1"],
            # title="Confidence Score Profile",
            labels={"value": "Value", "variable": "Metric", "scores": "Confidence Score"},
            width=None,
            height=500,
            color_discrete_map=color_map,
        )
        fig.update_traces(
            hovertemplate="Confidence Score: %{x:.2f}<br>Value: %{y:.2f}<extra></extra>"
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        if self._loader.mp.f1_optimal_conf is not None and self._loader.mp.best_f1 is not None:
            # Add vertical line for the best threshold
            fig.add_shape(
                type="line",
                x0=self._loader.mp.f1_optimal_conf,
                x1=self._loader.mp.f1_optimal_conf,
                y0=0,
                y1=self._loader.mp.best_f1,
                line=dict(color="gray", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=self._loader.mp.f1_optimal_conf,
                y=self._loader.mp.best_f1 + 0.04,
                text=f"F1-optimal threshold: {self._loader.mp.f1_optimal_conf:.2f}",
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
