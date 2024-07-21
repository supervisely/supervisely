from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ujson
from jinja2 import Template
from pycocotools.coco import COCO

from supervisely._utils import *
from supervisely.api.api import Api
from supervisely.collection.str_enum import StrEnum
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.io.fs import *
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import MetricProvider
from supervisely.nn.benchmark.metric_visualizations import *
from supervisely.nn.benchmark.metric_visualizations import MetricVis
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

_METRIC_VISUALIZATIONS = (
    # Overview,
    # ExplorerGrid,
    # ModelPredictions,
    # WhatIs,
    # OutcomeCounts,
    Recall,
    Precision,
    # RecallVsPrecision,
    # PRCurve,
    # PRCurveByClass,
    # ConfusionMatrix,
    # FrequentlyConfused,
    # IOUDistribution,
    # ReliabilityDiagram,
    # ConfidenceScore,
    # F1ScoreAtDifferentIOU,
    # ConfidenceDistribution,
    # PerClassAvgPrecision,
    # PerClassOutcomeCounts,
    # segmentation-only
    # # TODO integrate binary files while saving to self.tmp_dir to the current solution
    # OverallErrorAnalysis,
    # ClasswiseErrorAnalysis,
)


def generate_main_template(metric_visualizations: List[MetricVis]):
    template_str = """<div>
    <sly-iw-sidebar :options="{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false }">
        <div slot="sidebar">"""

    for vis in metric_visualizations:
        template_str += vis.template_sidebar_str

    template_str += """\n        </div>
      
        <div style="padding: 0 15px;">"""

    for vis in metric_visualizations:
        template_str += vis.template_main_str

    template_str += "\n        </div>\n    </sly-iw-sidebar>\n</div>"

    return template_str


class IdMapper:
    def __init__(self, coco_dataset: dict):
        self.map_img = {x["id"]: x["sly_id"] for x in coco_dataset["images"]}
        self.map_obj = {x["id"]: x["sly_id"] for x in coco_dataset["annotations"]}


class ClickData:
    def __init__(self, m: MetricProvider, gt_id_mapper: IdMapper, dt_id_mapper: IdMapper):
        self.m = m
        # self.m_full = m_full
        self.gt_id_mapper = gt_id_mapper
        self.dt_id_mapper = dt_id_mapper
        self.catId2name = {cat_id: cat["name"] for cat_id, cat in m.cocoGt.cats.items()}

        self.outcome_counts = {
            "TP": self._gather_matches(self.m.tp_matches),
            "FN": self._gather_matches(self.m.fn_matches),
            "FP": self._gather_matches(self.m.fp_matches),
        }

        outcome_counts_by_class = defaultdict(lambda: {"TP": [], "FN": [], "FP": []})
        for match in self.m.matches:
            cat_id = match["category_id"]
            cat_name = self.m.cocoGt.cats[cat_id]["name"]
            outcome_counts_by_class[cat_name][match["type"]].append(self._gather(match))
        self.outcome_counts_by_class = dict(outcome_counts_by_class)

        self.objects_by_class = {cat_name: [] for cat_name in self.m.cat_names}  # ! ??? ШТОЭТА???
        for match in self.m.matches:
            cat_id = match["category_id"]
            cat_name = self.m.cocoGt.cats[cat_id]["name"]
            self.objects_by_class[cat_name].append(self._gather(match))

        self.confusion_matrix = self._confusion_matrix()

        self.frequently_confused = self._frequently_confused(self.confusion_matrix)

    def _confusion_matrix(self):
        confusion_matrix_ids = defaultdict(list)
        none_name = "(None)"
        for match in self.m.confused_matches:
            cat_pred = self.catId2name[match["category_id"]]
            cat_gt = self.catId2name[self.m.cocoGt.anns[match["gt_id"]]["category_id"]]
            confusion_matrix_ids[(cat_pred, cat_gt)].append(self._gather(match))

        for match in self.m.tp_matches:
            cat_name = self.catId2name[match["category_id"]]
            confusion_matrix_ids[(cat_name, cat_name)].append(self._gather(match))

        for match in self.m.fp_not_confused_matches:
            cat_pred = self.catId2name[match["category_id"]]
            confusion_matrix_ids[(cat_pred, none_name)].append(self._gather(match))

        for match in self.m.fn_matches:
            cat_gt = self.catId2name[match["category_id"]]
            confusion_matrix_ids[(none_name, cat_gt)].append(self._gather(match))
        return confusion_matrix_ids

    def _frequently_confused(self, confusion_matrix_ids: dict):
        cm = self.m.confusion_matrix()
        fcp = self.m.frequently_confused(cm)
        pairs = fcp["category_pair"]
        frequently_confused = {}
        for i, pair in enumerate(pairs):
            cat_a, cat_b = pair
            joint = confusion_matrix_ids[(cat_a, cat_b)] + confusion_matrix_ids[(cat_b, cat_a)]
            joint = sorted(joint, key=lambda x: x["gt_img_id"])
            frequently_confused[pair] = joint
        return frequently_confused

    def _gather(self, match: dict):
        return {
            "gt_img_id": self.gt_id_mapper.map_img[match["image_id"]],
            "dt_img_id": self.dt_id_mapper.map_img[match["image_id"]],
            "gt_obj_id": self.gt_id_mapper.map_obj.get(match["gt_id"]),
            "dt_obj_id": self.dt_id_mapper.map_obj.get(match["dt_id"]),
        }

    def _gather_matches(self, matches: list):
        return [self._gather(d) for d in matches]


class MetricLoader:

    def __init__(self, cocoGt_path: str, cocoDt_path: str, eval_data_path: str) -> None:

        with open(cocoGt_path, "r") as f:
            cocoGt_dataset = json.load(f)
        with open(cocoDt_path, "r") as f:
            cocoDt_dataset = json.load(f)

        # Remove COCO read logs
        with HiddenCocoPrints():
            cocoGt = COCO()
            cocoGt.dataset = cocoGt_dataset
            cocoGt.createIndex()
            cocoDt = cocoGt.loadRes(cocoDt_dataset["annotations"])

        with open(eval_data_path, "rb") as f:
            eval_data = pickle.load(f)

        # mp = MetricProvider(
        #     matches,
        #     eval_data["coco_metrics"],
        #     eval_data["params"],
        #     cocoGt,
        #     cocoDt,
        # )
        # mp.calculate()

        # self.m = mp.m

        self.m_full = MetricProvider(
            eval_data["matches"],
            eval_data["coco_metrics"],
            eval_data["params"],
            cocoGt,
            cocoDt,
        )
        self.score_profile = self.m_full.confidence_score_profile()
        self.f1_optimal_conf, self.best_f1 = self.m_full.get_f1_optimal_conf()
        # print(f"F1-Optimal confidence: {self.f1_optimal_conf:.4f} with f1: {self.best_f1:.4f}")

        matches_thresholded = metric_provider.filter_by_conf(
            eval_data["matches"], self.f1_optimal_conf
        )
        self.m = MetricProvider(
            matches_thresholded, eval_data["coco_metrics"], eval_data["params"], cocoGt, cocoDt
        )
        self.df_score_profile = pd.DataFrame(
            self.score_profile, columns=["scores", "Precision", "Recall", "F1"]
        )

        self.per_class_metrics: pd.DataFrame = self.m.per_class_metrics()
        self.per_class_metrics_sorted: pd.DataFrame = self.per_class_metrics.sort_values(by="f1")

        # downsample
        if len(self.df_score_profile) > 5000:
            self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        else:
            self.dfsp_down = self.df_score_profile

        # Click data
        gt_id_mapper = IdMapper(cocoGt_dataset)
        dt_id_mapper = IdMapper(cocoDt_dataset)

        self.click_data = ClickData(self.m, gt_id_mapper, dt_id_mapper)

        self.base_metrics = self.m.base_metrics()

        self.tmp_dir = None

        self._api = Api.from_env()
        self.gt_project_id = 39099
        self.gt_dataset_id = 92810
        self.dt_project_id = 39141
        self.dt_dataset_id = 92872
        self.diff_project_id = 39249
        self.diff_dataset_id = 93099

        self.dt_project_info = self._api.project.get_info_by_id(
            self.dt_project_id, raise_error=True
        )
        self.dt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.dt_project_id)
        )

        datasets = self._api.dataset.get_list(self.dt_project_id)

        tmp = {}
        self.dt_images = {}
        self.dt_images_by_name = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            tmp[d.id] = [x.id for x in images]
            for info in images:
                self.dt_images[info.id] = info
                self.dt_images_by_name[info.name] = info

        self.dt_ann_jsons = {
            ann.image_id: ann.annotation
            for d in datasets
            for ann in self._api.annotation.download_batch(d.id, tmp[d.id])
        }

    def upload_layout(self, team_id: str, dest_dir: str):
        self.tmp_dir = f"/tmp/tmp{rand_str(10)}"
        mkdir(f"{self.tmp_dir}/data", remove_content_if_exists=True)

        initialized = [mv(self) for mv in _METRIC_VISUALIZATIONS]
        for new_mv in initialized:
            for widget in new_mv.schema:
                self._write_markdown_files(new_mv, widget)
                self._write_json_files(new_mv, widget)
        self._save_template(initialized)

        with tqdm_sly(
            desc="Uploading .json to teamfiles",
            total=get_directory_size(self.tmp_dir),
            unit="B",
            unit_scale=True,
        ) as pbar:
            self._api.file.upload_directory(
                team_id,
                self.tmp_dir,
                dest_dir,
                replace_if_conflict=True,
                change_name_if_conflict=False,
                progress_size_cb=pbar,
            )

        if dir_exists(self.tmp_dir):
            remove_dir(self.tmp_dir)

        logger.info(f"Uploaded to: {dest_dir!r}")

    def _write_markdown_files(self, metric_visualization: MetricVis, widget: Widget):

        if isinstance(widget, Widget.Markdown):
            content = metric_visualization.get_md_content(self, widget)
            local_path = f"{self.tmp_dir}/data/{widget.name}.md"
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info("Saved: %r", f"{widget.name}.md")

        if isinstance(widget, Widget.Collapse):
            for subwidget in widget.schema:
                content = metric_visualization.get_md_content(self, subwidget)
                local_path = f"{self.tmp_dir}/data/{subwidget.name}.md"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)

                logger.info("Saved: %r", f"{subwidget.name}.md")

    def _write_json_files(self, mv: MetricVis, widget: Widget):

        if isinstance(widget, Widget.Chart):
            fig = mv.get_figure(self, widget)
            if fig is not None:
                fig_data = {
                    "selected": None,
                    "galleryContent": "",
                    "dialogVisible": False,
                    "chartContent": json.loads(fig.to_json()),
                }
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.tmp_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(fig_data, f)
                logger.info("Saved: %r", basename)

                click_data = mv.get_click_data(self, widget)
                if click_data is not None:
                    basename = f"{widget.name}_{mv.name}_clickdata.json"
                    local_path = f"{self.tmp_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(ujson.dumps(click_data))
                    logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Gallery):
            content = mv.get_gallery(self, widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.tmp_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Table):
            content = mv.get_table(self, widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.tmp_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

    def _generate_template(self, metric_visualizations: Tuple[MetricVis]) -> str:
        html_snippets = {}
        main_template = Template(generate_main_template(metric_visualizations))
        for mv in metric_visualizations:
            for widget in mv.schema:
                if isinstance(widget, Widget.Notification):
                    html_snippets.update(mv.get_html_snippets(self))

            html_snippets.update(mv.get_html_snippets(self))

        return main_template.render(**html_snippets)

    def _generate_state(self, metric_visualizations: Tuple[MetricVis]) -> dict:
        res = {}
        for mv in metric_visualizations:
            for widget in mv.schema:
                if isinstance(widget, Widget.Chart) and mv.switchable:
                    res[mv.radiogroup_id] = widget.switch_key
                    break
        return res

    def _save_template(self, metric_visualizations: Tuple[MetricVis]):
        local_path = f"{self.tmp_dir}/template.vue"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(self._generate_template(metric_visualizations))
        logger.info("Saved: %r", "template.vue")
        local_path = f"{self.tmp_dir}/state.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(self._generate_state(metric_visualizations), f)
        logger.info("Saved: %r", "state.json")
