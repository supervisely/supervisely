from __future__ import annotations

from typing import Dict

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class OutcomeCounts(BaseVisMetric):
    MARKDOWN = "outcome_counts"
    CHART = "outcome_counts"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_outcome_counts.format(
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
            self.vis_texts.definitions.false_negatives,
        )
        return MarkdownWidget(self.MARKDOWN, "Outcome Counts", text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  # -> go.Figure:
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[self.eval_result.mp.TP_count],
                y=["Outcome"],
                name="TP",
                orientation="h",
                marker=dict(color="#8ACAA1"),
                hovertemplate="TP: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self.eval_result.mp.FN_count],
                y=["Outcome"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
                hovertemplate="FN: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self.eval_result.mp.FP_count],
                y=["Outcome"],
                name="FP",
                orientation="h",
                marker=dict(color="#F7ADAA"),
                hovertemplate="FP: %{x} objects<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="stack",
            width=600,
            height=300,
        )
        fig.update_xaxes(title_text="Count (objects)")
        fig.update_yaxes(tickangle=-90)

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

    def get_click_data(self) -> Dict:
        if not self.clickable:
            return
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for outcome, matches_data in self.eval_result.click_data.outcome_counts.items():
            res["clickData"][outcome] = {}
            res["clickData"][outcome]["imagesIds"] = []

            img_ids = set()
            for match_data in matches_data:
                pairs_data = self.eval_result.matched_pair_data[match_data["gt_img_id"]]
                if outcome == "FN":
                    img_ids.add(pairs_data.diff_image_info.id)
                else:
                    img_ids.add(pairs_data.pred_image_info.id)

            res["clickData"][outcome][
                "title"
            ] = f"{outcome}: {len(matches_data)} object{'s' if len(matches_data) > 1 else ''}"
            res["clickData"][outcome]["imagesIds"] = list(img_ids)
            res["clickData"][outcome]["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                {"type": "tag", "tagId": "outcome", "value": outcome},
            ]

        return res
