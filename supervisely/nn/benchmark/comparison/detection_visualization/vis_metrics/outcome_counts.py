from collections import defaultdict
from typing import List

import numpy as np

from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import ChartWidget
from supervisely.nn.task_type import TaskType


class OutcomeCounts(BaseVisMetrics):
    CHART_MAIN = "chart_outcome_counts"
    CHART_COMPARISON = "chart_outcome_counts_comparison"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.imgIds_to_anns = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.coco_to_sly_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(tuple)))

        self._initialize_ids_mapping()

        self.common_and_diff_tp = self._find_common_and_diff_tp()
        self.common_and_diff_fn = self._find_common_and_diff_fn()
        self.common_and_diff_fp = self._find_common_and_diff_fp()

    def _initialize_ids_mapping(self):
        for idx, r in enumerate(self.eval_results):
            l = {
                "TP": (r.mp.cocoDt.anns, r.mp.m.tp_matches, r.click_data.outcome_counts),
                "FN": (r.mp.cocoGt.anns, r.mp.m.fn_matches, r.click_data.outcome_counts),
                "FP": (r.mp.cocoDt.anns, r.mp.m.fp_matches, r.click_data.outcome_counts),
            }
            for outcome, (coco_anns, matches_data, sly_data) in l.items():
                for m, sly_m in zip(matches_data, sly_data[outcome]):
                    key = m["dt_id"] if outcome != "FN" else m["gt_id"]
                    gt_key = m["gt_id"] if outcome != "FP" else m["dt_id"]
                    ann = coco_anns[key]
                    self.imgIds_to_anns[idx][outcome][key].append(ann)
                    self.coco_to_sly_ids[idx][outcome][gt_key] = sly_m

    @property
    def chart_widget_main(self) -> ChartWidget:
        chart = ChartWidget(name=self.CHART_MAIN, figure=self.get_main_figure())
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_main_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].y}${'_'}${payload.points[0].data.name}`,",
        )
        return chart

    @property
    def chart_widget_comparison(self) -> ChartWidget:
        chart = ChartWidget(
            name=self.CHART_COMPARISON,
            figure=self.get_comparison_figure(),
        )
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_comparison_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].y}${'_'}${payload.points[0].data.name}`,",
        )
        return chart

    def _update_figure_layout(self, fig):
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
        model_names = [f"[{i}] {e.short_name}" for i, e in enumerate(self.eval_results, 1)][::-1]
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

        fig = self._update_figure_layout(fig)
        return fig

    def get_comparison_figure(self):  #  -> Optional[go.Figure]
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        colors = ["#8ACAA1", "#dd3f3f", "#F7ADAA"]
        model_names = [f"[{i}] {e.short_name}" for i, e in enumerate(self.eval_results, 1)][::-1]
        model_names.append("Common")

        diff_tps, common_tps = self.common_and_diff_tp
        diff_fns, common_fns = self.common_and_diff_fn
        diff_fps, common_fps = self.common_and_diff_fp
        tps_cnt = [len(x) for x in diff_tps[::-1]] + [len(common_tps)]
        fns_cnt = [len(x) for x in diff_fns[::-1]] + [len(common_fns)]
        fps_cnt = [len(x) for x in diff_fps[::-1]] + [len(common_fps)]

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

        fig = self._update_figure_layout(fig)
        return fig

    def _find_common_and_diff_fn(self) -> List[int]:
        ids = [
            dict([(x["gt_obj_id"], x["gt_img_id"]) for x in r.click_data.outcome_counts["FN"]])
            for r in self.eval_results
        ]
        same = set.intersection(*[set(x.keys()) for x in ids])
        diffs = [set(x.keys()) - same for x in ids]

        same = {i: ids[0][i] for i in same}
        diffs = [{i: s[i] for i in d} for s, d in zip(ids, diffs)]

        return diffs, same

    def _get_coco_key_name(self):
        task_type_to_key_name = {
            TaskType.OBJECT_DETECTION: "bbox",
            TaskType.INSTANCE_SEGMENTATION: "segmentation",
            TaskType.SEMANTIC_SEGMENTATION: "segmentation",
        }
        key_name = task_type_to_key_name.get(self.eval_results[0].cv_task)
        if key_name is None:
            raise NotImplementedError("Not implemented for this task type")
        return key_name

    def _find_common_and_diff_fp(self) -> List[int]:
        from pycocotools import mask as maskUtils  # pylint: disable=import-error

        iouThr = 0.75
        key_name = self._get_coco_key_name()

        imgIds_to_anns = [self.imgIds_to_anns[idx]["FP"] for idx in range(len(self.eval_results))]
        sly_ids_list = [
            {x["dt_obj_id"]: x["dt_img_id"] for x in r.click_data.outcome_counts["FP"]}
            for r in self.eval_results
        ]

        same_fp_matches = []
        for img_id in imgIds_to_anns[0]:
            anns_list = [imgIds[img_id] for imgIds in imgIds_to_anns]
            geoms_list = [[x[key_name] for x in anns] for anns in anns_list]

            if any(len(geoms) == 0 for geoms in geoms_list):
                continue

            ious_list = [
                maskUtils.iou(geoms_list[0], geoms, [0] * len(geoms)) for geoms in geoms_list[1:]
            ]
            if any(len(ious) == 0 for ious in ious_list):
                continue

            indxs_list = [np.nonzero(ious > iouThr) for ious in ious_list]
            if any(len(indxs[0]) == 0 for indxs in indxs_list):
                continue

            indxs_list = [list(zip(*indxs)) for indxs in indxs_list]
            indxs_list = [
                sorted(indxs, key=lambda x: ious[x[0], x[1]], reverse=True)
                for indxs, ious in zip(indxs_list, ious_list)
            ]

            id_sets = [set(idxs[0]) for idxs in indxs_list]
            common_ids = set.intersection(*id_sets)
            if not common_ids:
                continue

            for i, j in indxs_list[0]:
                if i in common_ids:
                    same_fp_matches.append((anns_list[0][i], [anns[j] for anns in anns_list[1:]]))
                    common_ids.remove(i)

        # Find different FP matches for each model
        same_fp_ids = set(x[0]["id"] for x in same_fp_matches)
        diff_fp_matches = [
            set([x["dt_id"] for x in eval_result.mp.m.fp_matches]) - same_fp_ids
            for eval_result in self.eval_results
        ]

        diff_fp_matches_dicts = []
        for idx, diff_fp in enumerate(diff_fp_matches):
            diff_fp_dict = {}
            for x in diff_fp:
                obj_id = self.coco_to_sly_ids[idx]["FP"][x]["dt_obj_id"]
                img_id = sly_ids_list[idx][obj_id]
                diff_fp_dict[obj_id] = img_id
            diff_fp_matches_dicts.append(diff_fp_dict)

        same_fp_matches_dict = {}
        for x in same_fp_matches:
            obj_id = self.coco_to_sly_ids[0]["FP"][x[0]["id"]]["dt_obj_id"]
            img_id = sly_ids_list[0][obj_id]
            same_fp_matches_dict[obj_id] = img_id

        return diff_fp_matches_dicts, same_fp_matches_dict

    def _find_common_and_diff_tp(self) -> tuple:

        ids = [
            dict([(x["gt_obj_id"], x) for x in r.click_data.outcome_counts["TP"]])
            for r in self.eval_results
        ]

        same = set.intersection(*[set(x.keys()) for x in ids])
        diffs = [set(x.keys()) - same for x in ids]

        same = {s["dt_obj_id"]: s["dt_img_id"] for s in [ids[0][i] for i in same]}
        diffs = [{s[i]["dt_obj_id"]: s[i]["dt_img_id"] for i in d} for s, d in zip(ids, diffs)]

        return diffs, same

    def get_main_click_data(self):
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for i, eval_result in enumerate(self.eval_results, 1):
            model_name = f"[{i}] {eval_result.name}"
            for outcome, matches_data in eval_result.click_data.outcome_counts.items():
                key = f"{model_name}_{outcome}"
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
                thr = eval_result.mp.conf_threshold
                if outcome == "FN":
                    outcome_dict["filters"] = [
                        {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
                    ]
                else:
                    outcome_dict["filters"] = [
                        {"type": "tag", "tagId": "outcome", "value": outcome},
                        {"type": "tag", "tagId": "confidence", "value": [thr, 1]},
                    ]

        return res

    def get_comparison_click_data(self):
        res = {}

        res["layoutTemplate"] = [None, None, None]

        res["clickData"] = {}

        outcomes_ids = {
            "TP": self.common_and_diff_tp,
            "FN": self.common_and_diff_fn,
            "FP": self.common_and_diff_fp,
        }

        def _update_outcome_dict(title, outcome, outcome_dict, ids):
            img_ids = set()
            obj_ids = set()
            for obj_id, img_id in ids.items():
                img_ids.add(img_id)
                obj_ids.add(obj_id)

            title = f"{title}. {outcome}: {len(obj_ids)} object{'s' if len(obj_ids) > 1 else ''}"
            outcome_dict["title"] = title
            outcome_dict["imagesIds"] = list(img_ids)
            filters = outcome_dict.setdefault("filters", [])
            filters.append({"type": "specific_objects", "tagId": None, "value": list(obj_ids)})
            if outcome != "FN":
                filters.append({"type": "tag", "tagId": "confidence", "value": [0, 1]})
                filters.append({"type": "tag", "tagId": "outcome", "value": outcome})

        for outcome, (diff_ids, common_ids) in outcomes_ids.items():
            key = f"Common_{outcome}"
            outcome_dict = res["clickData"].setdefault(key, {})

            _update_outcome_dict("Common", outcome, outcome_dict, common_ids)

            for i, diff_ids in enumerate(diff_ids, 1):
                name = f"[{i}] {self.eval_results[i - 1].name}"
                key = f"{name}_{outcome}"
                outcome_dict = res["clickData"].setdefault(key, {})

                _update_outcome_dict(name, outcome, outcome_dict, diff_ids)

        return res
