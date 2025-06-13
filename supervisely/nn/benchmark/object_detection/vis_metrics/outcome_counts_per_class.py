from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    ContainerWidget,
    MarkdownWidget,
    RadioGroupWidget,
)


class PerClassOutcomeCounts(DetectionVisMetric):
    MARKDOWN = "per_class_outcome_counts"
    MARKDOWN_2 = "per_class_outcome_counts_2"
    CHART = "per_class_outcome_counts"
    COLLAPSE_TIP = "per_class_outcome_counts_collapse"
    RADIO_GROUP = "per_class_outcome_counts_radio_group"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True
        self._keypair_sep: str = "-"
        self.switchable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_class_outcome_counts_1
        text = text.format(
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
            self.vis_texts.definitions.false_negatives,
        )
        return MarkdownWidget(self.MARKDOWN, "Outcome Counts by Class", text)

    @property
    def md_2(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_class_outcome_counts_2
        text = text.format(self.vis_texts.definitions.f1_score)
        return MarkdownWidget(self.MARKDOWN_2, "Outcome Counts by Class", text)

    @property
    def collapse(self) -> CollapseWidget:
        md = MarkdownWidget(
            name=self.COLLAPSE_TIP,
            title="Normalization",
            text=self.vis_texts.markdown_normalization,
        )
        return CollapseWidget([md])

    @property
    def chart(self) -> ContainerWidget:
        return ContainerWidget(
            [self.radio_group(), self._get_chart("normalized"), self._get_chart("absolute")],
            self.CHART,
        )

    def radio_group(self) -> RadioGroupWidget:
        return RadioGroupWidget(
            "Normalization",
            self.RADIO_GROUP,
            ["normalized", "absolute"],
        )

    def _get_chart(self, switch_key: Literal["normalized", "absolute"]) -> ChartWidget:
        chart = ChartWidget(
            self.CHART,
            self._get_figure(switch_key),
            switch_key=switch_key,
            switchable=self.switchable,
            radiogroup_id=self.RADIO_GROUP,
        )
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].label}${'-'}${payload.points[0].data.name}`, 'keySeparator': '-',",
        )
        return chart

    def _get_figure(self, switch_key: Literal["normalized", "absolute"]):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        # Per-class Counts
        mp = self.eval_result.mp
        tp = mp.m._take_iou_thresholds(mp.true_positives).flatten()
        fp = mp.m._take_iou_thresholds(mp.false_positives).flatten()
        fn = mp.m._take_iou_thresholds(mp.false_negatives).flatten()

        # normalize
        support = tp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_rel = tp / support
            fp_rel = fp / support
            fn_rel = fn / support

            # sort by f1
            sort_scores = 2 * tp / (2 * tp + fp + fn)

        K = len(self.eval_result.mp.cat_names)
        sort_indices = np.argsort(sort_scores)
        cat_names_sorted = [self.eval_result.mp.cat_names[i] for i in sort_indices]
        tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

        objects_count = np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]])
        data = {
            "Type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
            "category": cat_names_sorted * 3,
        }
        y_label = ""
        if switch_key == "normalized":
            y_label = "Objects Fraction"
            data["count"] = np.concatenate([tp_rel, fn_rel, fp_rel])
        elif switch_key == "absolute":
            y_label = "Objects Count"
            data["count"] = objects_count

        df = pd.DataFrame(data)

        color_map = {"TP": "#8ACAA1", "FN": "#dd3f3f", "FP": "#F7ADAA"}
        fig = px.bar(
            df,
            x="category",
            y="count",
            color="Type",
            height=500,
            width=1000,
            labels={"count": y_label, "category": "Class"},
            color_discrete_map=color_map,
        )
        xaxis_title = fig.layout.xaxis.title.text
        yaxis_title = fig.layout.yaxis.title.text
        if switch_key == "normalized":

            fig.update_traces(
                hovertemplate="Type=%{fullData.name} <br>"
                + xaxis_title
                + "=%{x}<br>"
                + yaxis_title
                + "=%{y:.2f}<extra></extra>"
            )
        elif switch_key == "absolute":
            fig.update_traces(
                hovertemplate="Type=%{fullData.name} <br>"
                + xaxis_title
                + "=%{x}<br>"
                + yaxis_title
                + "=%{y}<extra></extra>",
            )
        return fig

    def get_click_data(self) -> Dict:
        if not self.clickable:
            return
        res = {}
        res["layoutTemplate"] = [None, None, None]

        res["clickData"] = {}
        for class_name, v1 in self.eval_result.click_data.outcome_counts_by_class.items():
            for outcome, matches_data in v1.items():
                key = class_name + self._keypair_sep + outcome
                res["clickData"][key] = {}
                res["clickData"][key]["imagesIds"] = []
                res["clickData"][key][
                    "title"
                ] = f"Images with objects of class '{class_name}' and outcome '{outcome}'"

                img_ids = set()
                for match_data in matches_data:
                    img_comparison_data = self.eval_result.matched_pair_data[
                        match_data["gt_img_id"]
                    ]
                    if outcome == "FN":
                        img_ids.add(img_comparison_data.diff_image_info.id)
                    else:
                        img_ids.add(img_comparison_data.pred_image_info.id)
                res["clickData"][key]["imagesIds"] = list(img_ids)
                res["clickData"][key]["filters"] = [
                    {
                        "type": "tag",
                        "tagId": "confidence",
                        "value": [self.eval_result.mp.conf_threshold, 1],
                    },
                    {"type": "tag", "tagId": "outcome", "value": outcome},
                ]
        return res
