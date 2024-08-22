from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_texts import definitions
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class OutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable: bool = True
        self.schema = Schema(
            markdown_outcome_counts=Widget.Markdown(
                title="Outcome Counts",
                is_header=True,
                formats=[
                    definitions.true_positives,
                    definitions.false_positives,
                    definitions.false_negatives,
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
        for key, v in self._loader.click_data.outcome_counts.items():
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []

            img_ids = set()
            for x in v:
                if key == "FN":
                    dt_image = self._loader.dt_images_dct[x["dt_img_id"]]
                    img_ids.add(self._loader.diff_images_dct_by_name[dt_image.name].id)
                else:
                    img_ids.add(self._loader.dt_images_dct[x["dt_img_id"]].id)

            res["clickData"][key]["title"] = f"{key}: {len(v)} object{'s' if len(v) > 1 else ''}"
            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                {"type": "tag", "tagId": "outcome", "value": key},
            ]

        return res
