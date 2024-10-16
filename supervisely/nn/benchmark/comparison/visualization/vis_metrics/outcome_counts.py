from collections import defaultdict
from typing import List

import numpy as np

from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import ChartWidget
from supervisely.nn.task_type import TaskType


class OutcomeCounts(BaseVisMetric):

    CHART_MAIN = "chart_outcome_counts"
    CHART_COMPARISON = "chart_outcome_counts_comparison"

    @property
    def chart_widget_main(self) -> ChartWidget:
        chart = ChartWidget(name=self.CHART_MAIN, figure=self.get_main_figure())
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_main_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].data.name}`,",
        )
        return chart

    @property
    def chart_widget_comparison(self) -> ChartWidget:
        chart = ChartWidget(name=self.CHART_COMPARISON, figure=self.get_comparison_figure(),)
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_comparison_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].data.name}`,",
        )

    def update_figure_layout(self, fig):
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

    def get_main_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        tp_counts = [eval_result.mp.TP_count for eval_result in self.eval_results][::-1]
        fn_counts = [eval_result.mp.FN_count for eval_result in self.eval_results][::-1]
        fp_counts = [eval_result.mp.FP_count for eval_result in self.eval_results][::-1]
        model_names = [f"Model {idx}" for idx in range(1, len(self.eval_results) + 1)][::-1]
        counts = [tp_counts, fn_counts, fp_counts]
        names = ["TP", "FN", "FP"]
        colors = ["#8ACAA1", "#dd3f3f", "#F7ADAA"]

        for metric, values, color in zip(names, counts, colors):
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=model_names,
                    name=metric,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate=f"{metric}: %{{x}} objects<extra></extra>",
                )
            )

        fig = self.update_figure_layout(fig)
        return fig

    def get_comparison_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go

        fig = go.Figure()

        colors = ["#8ACAA1", "#dd3f3f", "#F7ADAA"]
        model_names = [f"Model {idx}" for idx in range(1, len(self.eval_results) + 1)][::-1]
        model_names.append("Common")

        tps_cnt = [len(x) for x in self.find_common_and_diff_tp()]
        fns_cnt = [len(x) for x in self.find_common_and_diff_fn()]
        fps_cnt = [len(x) for x in self.find_common_and_diff_fp()]

        for metric, values, color in zip(["TP", "FN", "FP"], [tps_cnt, fns_cnt, fps_cnt], colors):
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=model_names,
                    name=metric,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate=f"{metric}: %{{x}} objects<extra></extra>",
                )
            )

        fig = self.update_figure_layout(fig)
        return fig

    def find_common_and_diff_fn(self) -> List[int]:
        ids_list = [set([x["gt_id"] for x in r.mp.m.fn_matches]) for r in self.eval_results]

        same_fn_matches = set.intersection(*ids_list)
        diff_fn_matches = [ids - same_fn_matches for ids in ids_list]
        return diff_fn_matches[::-1] + [same_fn_matches]

    def find_common_and_diff_fp(self) -> List[int]:
        from pycocotools import mask as maskUtils  # pylint: disable=import-error

        iouThr = 0.75

        if self.eval_results[0].cv_task == TaskType.OBJECT_DETECTION:
            key_name = "bbox"
        elif self.eval_results[0].cv_task in [
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]:
            key_name = "segmentation"
        else:
            raise NotImplementedError("Not implemented for this task type")

        # TODO: add support for more models
        assert len(self.eval_results) == 2, "Currently only 2 models are supported"

        imgId2ann1 = defaultdict(list)
        imgId2ann2 = defaultdict(list)
        for m in self.eval_results[0].mp.m.fp_matches:
            ann = self.eval_results[0].mp.cocoDt.anns[m["dt_id"]]
            imgId2ann1[m["image_id"]].append(ann)
        for m in self.eval_results[1].mp.m.fp_matches:
            ann = self.eval_results[1].mp.cocoDt.anns[m["dt_id"]]
            imgId2ann2[m["image_id"]].append(ann)

        same_fp_matches = []
        for img_id in imgId2ann1:
            anns1 = imgId2ann1[img_id]
            anns2 = imgId2ann2[img_id]
            geoms1 = [x[key_name] for x in anns1]
            geoms2 = [x[key_name] for x in anns2]

            ious = maskUtils.iou(geoms1, geoms2, [0] * len(geoms2))
            if len(ious) == 0:
                continue
            indxs = np.nonzero(ious > iouThr)
            if len(indxs[0]) == 0:
                continue
            indxs = list(zip(*indxs))
            indxs = sorted(indxs, key=lambda x: ious[x[0], x[1]], reverse=True)
            id1, id2 = list(zip(*indxs))
            id1, id2 = set(id1), set(id2)
            for i, j in indxs:
                if i in id1 and j in id2:
                    same_fp_matches.append((anns1[i], anns2[j], ious[i, j]))
                    id1.remove(i)
                    id2.remove(j)

        # Find different FP matches for each model
        id1, id2 = zip(*[(x[0]["id"], x[1]["id"]) for x in same_fp_matches])
        id1 = set(id1)
        id2 = set(id2)

        diff_fp_matches_1 = set([x["dt_id"] for x in self.eval_results[0].mp.m.fp_matches]) - id1
        diff_fp_matches_2 = set([x["dt_id"] for x in self.eval_results[1].mp.m.fp_matches]) - id2

        return [diff_fp_matches_2, diff_fp_matches_1, same_fp_matches]

    def find_common_and_diff_tp(self) -> List[int]:
        ids_list = [set([x["gt_id"] for x in r.mp.m.tp_matches]) for r in self.eval_results]

        same_tp_matches = set.intersection(*ids_list)
        diff_tp_matches = [ids - same_tp_matches for ids in ids_list]

        return diff_tp_matches[::-1] + [same_tp_matches]

    def get_main_click_data(self):
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for i, eval_result in enumerate(self.eval_results):
            model_name = f"Model {i+1}"
            for outcome, matches_data in eval_result.click_data.outcome_counts.items():
                key = f"{i}_{outcome}"
                outcome_dict = res["clickData"].setdefault(key, {})
                outcome_dict["imagesIds"] = []

                img_ids = set()
                obj_ids = set()
                for x in matches_data:
                    img_ids.add(x["dt_img_id"] if outcome != "FN" else x["gt_img_id"])
                    obj_ids.add(x["dt_obj_id"] if outcome != "FN" else x["gt_obj_id"])

                title = f"{model_name}. {outcome}: {len(obj_ids)} object{'s' if len(obj_ids) > 1 else ''}"
                outcome_dict["title"] = title
                outcome_dict["imagesIds"] = list(img_ids)
                outcome_dict["filters"] = [
                    {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                    {"type": "tag", "tagId": "outcome", "value": outcome},
                    {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
                ]

        return res

    def get_comparison_click_data(self):
        res = {}

        res["layoutTemplate"] = [None, None, None]

        res["clickData"] = {}

        tp_matches = self.find_common_and_diff_tp()
        fn_matches = self.find_common_and_diff_fn()
        fp_matches = self.find_common_and_diff_fp()
        outcomes_matches = {"TP": tp_matches, "FN": fn_matches, "FP": fp_matches}

        common_img_ids = defaultdict(set)
        common_obj_ids = defaultdict(set)
        for i, eval_result in enumerate(self.eval_results):
            model_name = f"Model {i+1}"
            for outcome, matches_data in eval_result.click_data.outcome_counts.items():
                key = f"{i}_{outcome}"
                outcome_dict = res["clickData"].setdefault(key, {})
                outcome_dict["imagesIds"] = []

                compared_matches = outcomes_matches[outcome][i]
                img_ids = set()
                obj_ids = set()
                for x in matches_data:
                    img_id = x["dt_img_id"] if outcome != "FN" else x["gt_img_id"]
                    obj_id = x["dt_obj_id"] if outcome != "FN" else x["gt_obj_id"]
                    if img_id in compared_matches:
                        img_ids.add(img_id)
                        obj_ids.add(obj_id)
                    if img_id in outcomes_matches[outcome][-1]:
                        common_img_ids[outcome].add(img_id)
                        common_obj_ids[outcome].add(obj_id)

                title = f"{model_name}. {outcome}: {len(obj_ids)} object{'s' if len(obj_ids) > 1 else ''}"
                outcome_dict["title"] = title
                outcome_dict["imagesIds"] = list(img_ids)
                outcome_dict["filters"] = [
                    {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                    {"type": "tag", "tagId": "outcome", "value": outcome},
                    {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
                ]

        # Add common matches
        for outcome in ["TP", "FN", "FP"]:
            key = f"{len(self.eval_results)}_{outcome}"
            outcome_dict = res["clickData"].setdefault(key, {})
            obj_ids = list(common_obj_ids[outcome])
            outcome_dict["imagesIds"] = list(common_img_ids[outcome])
            title = f"Common {outcome}: {len(obj_ids)} object{'s' if len(obj_ids) > 1 else ''}"
            outcome_dict["title"] = title
            outcome_dict["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                {"type": "tag", "tagId": "outcome", "value": outcome},
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
