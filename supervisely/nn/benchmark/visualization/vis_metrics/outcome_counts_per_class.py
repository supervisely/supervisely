from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class PerClassOutcomeCounts(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.clickable: bool = True
        self.switchable: bool = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_class_outcome_counts_1=Widget.Markdown(
                title="Outcome Counts by Class",
                is_header=True,
                formats=[
                    self._loader.vis_texts.definitions.true_positives,
                    self._loader.vis_texts.definitions.false_positives,
                    self._loader.vis_texts.definitions.false_negatives,
                ],
            ),
            markdown_class_outcome_counts_2=Widget.Markdown(
                formats=[self._loader.vis_texts.definitions.f1_score]
            ),
            collapse_perclass_outcome=Widget.Collapse(
                Schema(
                    self._loader.vis_texts,
                    markdown_normalization=Widget.Markdown(title="Normalization"),
                )
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

        objects_count = np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]])
        data = {
            "Type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
            "category": cat_names_sorted * 3,
        }
        y_label = ""
        if widget.switch_key == "normalized":
            y_label = "Objects Fraction"
            # Stacked per-class counts
            data["count"] = np.concatenate([tp_rel, fn_rel, fp_rel])
        elif widget.switch_key == "absolute":
            y_label = "Objects Count"
            data["count"] = objects_count

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
        for class_name, v1 in self._loader.click_data.outcome_counts_by_class.items():
            for outcome, matches_data in v1.items():
                key = class_name + self._keypair_sep + outcome
                res["clickData"][key] = {}
                res["clickData"][key]["imagesIds"] = []
                res["clickData"][key][
                    "title"
                ] = f"Images with objects of class '{class_name}' and outcome '{outcome}'"

                img_ids = set()
                for match_data in matches_data:
                    img_comparison_data = self._loader.comparison_data[match_data["gt_img_id"]]
                    if outcome == "FN":
                        img_ids.add(img_comparison_data.diff_image_info.id)
                    else:
                        img_ids.add(img_comparison_data.pred_image_info.id)
                res["clickData"][key]["imagesIds"] = list(img_ids)
                res["clickData"][key]["filters"] = [
                    {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                    {"type": "tag", "tagId": "outcome", "value": outcome},
                ]
        return res
