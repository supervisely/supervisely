from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class OutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable: bool = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_outcome_counts=Widget.Markdown(
                title="Outcome Counts",
                is_header=True,
                formats=[
                    self._loader.vis_texts.definitions.true_positives,
                    self._loader.vis_texts.definitions.false_positives,
                    self._loader.vis_texts.definitions.false_negatives,
                ],
            ),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.TP_count],
                y=["Outcome"],
                name="TP",
                orientation="h",
                marker=dict(color="#8ACAA1"),
                hovertemplate="TP: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.FN_count],
                y=["Outcome"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
                hovertemplate="FN: %{x} objects<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[self._loader.mp.FP_count],
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

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable:
            return
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for outcome, matches_data in self._loader.click_data.outcome_counts.items():
            res["clickData"][outcome] = {}
            res["clickData"][outcome]["imagesIds"] = []

            img_ids = set()
            for match_data in matches_data:
                img_comparison_data = self._loader.comparison_data[match_data["gt_img_id"]]
                if outcome == "FN":
                    img_ids.add(img_comparison_data.diff_image_info.id)
                else:
                    img_ids.add(img_comparison_data.pred_image_info.id)

            res["clickData"][outcome][
                "title"
            ] = f"{outcome}: {len(matches_data)} object{'s' if len(matches_data) > 1 else ''}"
            res["clickData"][outcome]["imagesIds"] = list(img_ids)
            res["clickData"][outcome]["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                {"type": "tag", "tagId": "outcome", "value": outcome},
            ]

        return res
