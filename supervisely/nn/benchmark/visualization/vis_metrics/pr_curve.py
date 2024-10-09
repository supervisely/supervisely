from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class PRCurve(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_pr_curve=Widget.Markdown(
                title="Precision-Recall Curve",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.f1_score],
            ),
            collapse_pr=Widget.Collapse(
                schema=Schema(
                    self._loader.vis_texts,
                    markdown_trade_offs=Widget.Markdown(
                        title="About Trade-offs between precision and recall"
                    ),
                    markdown_what_is_pr_curve=Widget.Markdown(
                        title="How the PR curve is built?",
                        formats=[
                            self._loader.vis_texts.definitions.confidence_score,
                            self._loader.vis_texts.definitions.true_positives,
                            self._loader.vis_texts.definitions.false_positives,
                        ],
                    ),
                )
            ),
            notification_ap=Widget.Notification(
                formats_title=[loader.base_metrics()["mAP"].round(2)]
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Precision-Recall curve
        pr_curve = self._loader.mp.pr_curve().copy()
        pr_curve[pr_curve == -1] = np.nan  # -1 is a placeholder for no GT
        pr_curve = np.nanmean(pr_curve, axis=-1)
        fig = px.line(
            x=self._loader.mp.recThrs,
            y=pr_curve,
            # title="Precision-Recall Curve",
            labels={"x": "Recall", "y": "Precision"},
            width=600,
            height=500,
        )
        fig.data[0].name = "Model"
        fig.data[0].showlegend = True
        fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))
        fig.add_trace(
            go.Scatter(
                x=self._loader.mp.recThrs,
                y=[1] * len(self._loader.mp.recThrs),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
        fig.add_annotation(
            text=f"mAP = {self._loader.mp.base_metrics()['mAP']:.2f}",
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
        # fig.show()
        return fig
