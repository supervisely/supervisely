from __future__ import annotations

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class F1ScoreAtDifferentIOU(DetectionVisMetric):
    MARKDOWN = "f1_score_at_iou"
    CHART = "f1_score_at_iou"

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            self.MARKDOWN,
            "Confidence Profile at Different IoU thresholds",
            self.vis_texts.markdown_f1_at_ious,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        f1s = self.eval_result.mp.m_full.score_profile_f1s

        # downsample
        if len(self.eval_result.df_score_profile) > 5000:
            f1s_down = f1s[:, :: f1s.shape[1] // 1000]
        else:
            f1s_down = f1s

        iou_names = list(map(lambda x: str(round(x, 2)), self.eval_result.mp.iouThrs.tolist()))
        df = pd.DataFrame(
            np.concatenate([self.eval_result.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
            columns=["scores"] + iou_names,
        )
        labels = {"value": "F1-score", "variable": "IoU threshold", "scores": "Confidence Score"}

        fig = px.line(
            df,
            x="scores",
            y=iou_names,
            labels=labels,
            color_discrete_sequence=px.colors.sequential.Viridis,
            width=None,
            height=500,
        )
        fig.update_traces(
            hovertemplate="Confidence Score: %{x:.2f}<br>F1-score: %{y:.2f}<extra></extra>"
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

        # add annotations for maximum F1-Score for each IoU threshold
        for i, iou in enumerate(iou_names):
            # Skip if all F1 scores are NaN for this IoU threshold
            if np.isnan(f1s[i]).all():
                continue
            argmax_f1 = np.nanargmax(f1s[i])
            max_f1 = f1s[i][argmax_f1]
            score = self.eval_result.mp.m_full.score_profile["scores"][argmax_f1]
            fig.add_annotation(
                x=score,
                y=max_f1,
                text=f"Best conf: {score:.2f}",
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
