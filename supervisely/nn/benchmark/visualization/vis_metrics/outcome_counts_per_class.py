from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_texts import definitions
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class PerClassOutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable: bool = True
        self.switchable: bool = True
        self.schema = Schema(
            markdown_class_outcome_counts_1=Widget.Markdown(
                title="Outcome Counts by Class",
                is_header=True,
                formats=[
                    definitions.true_positives,
                    definitions.false_positives,
                    definitions.false_negatives,
                ],
            ),
            markdown_class_outcome_counts_2=Widget.Markdown(formats=[definitions.f1_score]),
            collapse_perclass_outcome=Widget.Collapse(
                Schema(markdown_normalization=Widget.Markdown(title="Normalization"))
            ),
            chart_01=Widget.Chart(switch_key="normalized"),
            chart_02=Widget.Chart(switch_key="absolute"),
        )

    def get_figure(self, widget: Widget.Chart):  #  -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        # Per-class Counts
        iou_thres = 0

        tp = self._loader.mp.true_positives[:, iou_thres]
        fp = self._loader.mp.false_positives[:, iou_thres]
        fn = self._loader.mp.false_negatives[:, iou_thres]

        # normalize
        support = tp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_rel = tp / support
            fp_rel = fp / support
            fn_rel = fn / support

            # sort by f1
            sort_scores = 2 * tp / (2 * tp + fp + fn)

        K = len(self._loader.mp.cat_names)
        sort_indices = np.argsort(sort_scores)
        cat_names_sorted = [self._loader.mp.cat_names[i] for i in sort_indices]
        tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

        images_count = np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]])
        data = {
            "Type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
            "category": cat_names_sorted * 3,
        }
        y_label = ""
        if widget.switch_key == "normalized":
            y_label = "Images Fraction"
            # Stacked per-class counts
            data["count"] = np.concatenate([tp_rel, fn_rel, fp_rel])
        elif widget.switch_key == "absolute":
            y_label = "Images Count"
            data["count"] = images_count

        df = pd.DataFrame(data)

        color_map = {"TP": "#8ACAA1", "FN": "#dd3f3f", "FP": "#F7ADAA"}
        fig = px.bar(
            df,
            x="category",
            y="count",
            color="Type",
            # title="Per-class Outcome Counts",
            height=500,
            width=1000,
            labels={"count": y_label, "category": "Class"},
            color_discrete_map=color_map,
        )
        xaxis_title = fig.layout.xaxis.title.text
        yaxis_title = fig.layout.yaxis.title.text
        if widget.switch_key == "normalized":

            fig.update_traces(
                hovertemplate="Type=%{fullData.name} <br>"
                + xaxis_title
                + "=%{x}<br>"
                + yaxis_title
                + "=%{y:.2f}<extra></extra>"
                # "Images count=%{y:.2f}<extra></extra>"
            )
        elif widget.switch_key == "absolute":
            fig.update_traces(
                hovertemplate="Type=%{fullData.name} <br>"
                + xaxis_title
                + "=%{x}<br>"
                + yaxis_title
                + "=%{y}<extra></extra>",
            )
        return fig

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        if not self.clickable:
            return
        res = {}
        res["layoutTemplate"] = [None, None, None]

        res["clickData"] = {}
        for key1, v1 in self._loader.click_data.outcome_counts_by_class.items():
            for key2, v2 in v1.items():
                key = key1 + self._keypair_sep + key2
                res["clickData"][key] = {}
                res["clickData"][key]["imagesIds"] = []

                res["clickData"][key]["title"] = f"Images with objects of class '{key1}' and outcome '{key2}'"

                img_ids = set()
                for x in v2:
                    if key2 == "FN":
                        dt_image = self._loader.dt_images_dct[x["dt_img_id"]]
                        img_ids.add(self._loader.diff_images_dct_by_name[dt_image.name].id)
                    else:
                        img_ids.add(self._loader.dt_images_dct[x["dt_img_id"]].id)
                res["clickData"][key]["imagesIds"] = list(img_ids)
                res["clickData"][key]["filters"] = [
                    {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                    {"type": "tag", "tagId": "outcome", "value": key2},
                ]
        return res
