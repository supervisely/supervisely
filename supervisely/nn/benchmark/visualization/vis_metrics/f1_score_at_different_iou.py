from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class F1ScoreAtDifferentIOU(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_f1_at_ious=Widget.Markdown(
                title="Confidence Profile at Different IoU thresholds",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.iou_threshold],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        # score_profile = self._loader.m_full.confidence_score_profile()
        f1s = self._loader.mp.m_full.score_profile_f1s

        # downsample
        if len(self._loader.df_score_profile) > 5000:
            f1s_down = f1s[:, :: f1s.shape[1] // 1000]
        else:
            f1s_down = f1s

        iou_names = list(map(lambda x: str(round(x, 2)), self._loader.mp.iouThrs.tolist()))
        df = pd.DataFrame(
            np.concatenate([self._loader.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
            columns=["scores"] + iou_names,
        )
        labels = {"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"}

        fig = px.line(
            df,
            x="scores",
            y=iou_names,
            # title="F1-Score at different IoU Thresholds",
            labels=labels,
            color_discrete_sequence=px.colors.sequential.Viridis,
            width=None,
            height=500,
        )
        fig.update_traces(
            hovertemplate="Confidence Score: %{x:.2f}<br>Value: %{y:.2f}<extra></extra>"
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        # add annotations for maximum F1-Score for each IoU threshold
        for i, iou in enumerate(iou_names):
            argmax_f1 = f1s[i].argmax()
            max_f1 = f1s[i][argmax_f1]
            score = self._loader.mp.m_full.score_profile["scores"][argmax_f1]
            fig.add_annotation(
                x=score,
                y=max_f1,
                text=f"Best score: {score:.2f}",
                showarrow=True,
                arrowhead=1,
                arrowcolor="black",
                ax=0,
                ay=-30,
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
