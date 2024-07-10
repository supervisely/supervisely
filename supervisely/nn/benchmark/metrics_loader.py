from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pycocotools.coco import COCO

import supervisely as sly
from src.click_data import ClickData
from src.utils import IdMapper
from supervisely._utils import *
from supervisely.api.api import Api
from supervisely.collection.str_enum import StrEnum
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.io.fs import *
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import MetricProvider
from supervisely.nn.benchmark.metrics import *
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

_METRICS = (
    Overview,
    OutcomeCounts,
    Recall,
    Precision,
    RecallVsPrecision,
    PRCurve,
    PRCurveByClass,
    ConfusionMatrix,
    FrequentlyConfused,
    IOUDistribution,
    ReliabilityDiagram,
    ConfidenceScore,
    ConfidenceDistribution,
    F1ScoreAtDifferentIOU,
    PerClassAvgPrecision,
    PerClassOutcomeCounts,
    # segmentation-only
    # # TODO integrate binary files while saving to self.tmp_dir to the current solution
    # OverallErrorAnalysis,
    # ClasswiseErrorAnalysis,
)


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
        self.outcome_counts = None
        self.outcome_counts_by_class = None
        self.objects_by_class = None
        self.confusion_matrix = None
        self.frequently_confused = None

    def create_data(self):
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


class MetricsLoader:

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

        self.m_full = metric_provider.MetricProvider(
            eval_data["matches"],
            eval_data["coco_metrics"],
            eval_data["params"],
            cocoGt,
            cocoDt,
        )
        self.score_profile = self.m_full.confidence_score_profile()
        self.f1_optimal_conf, self.best_f1 = self.m_full.get_f1_optimal_conf()
        print(f"F1-Optimal confidence: {self.f1_optimal_conf:.4f} with f1: {self.best_f1:.4f}")

        matches_thresholded = metric_provider.filter_by_conf(
            eval_data["matches"], self.f1_optimal_conf
        )
        self.m = metric_provider.MetricProvider(
            matches_thresholded, eval_data["coco_metrics"], eval_data["params"], cocoGt, cocoDt
        )
        self.df_score_profile = pd.DataFrame(self.score_profile)
        self.df_score_profile.columns = ["scores", "Precision", "Recall", "F1"]

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
        self.click_data.create_data()

        self.tmp_dir = f"/tmp/{rand_str(10)}"

    def upload_to(self, team_id: str, dest_dir: str):

        api = Api.from_env()

        for metric in _METRICS:
            fig = metric.get_figure(self)
            if fig is not None:
                self._write_fig(metric, fig)
            figs = metric.get_switchable_figures(self)
            if figs is not None:
                for idx, fig in enumerate(figs, start=1):
                    self._write_fig(metric, fig, fig_idx=idx)

        table_preds = self.m.prediction_table()
        basename = "prediction_table.json"
        local_path = f"{self.tmp_dir}/{basename}"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(table_preds.to_json())
        logger.info("Saved: %r", basename)

        with tqdm_sly(
            desc="Uploading .json to teamfiles",
            total=get_directory_size(self.tmp_dir),
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.upload_directory(
                team_id,
                self.tmp_dir,
                dest_dir,
                replace_if_conflict=True,
                change_name_if_conflict=False,
                progress_size_cb=pbar,
            )

        logger.info("Done.")

    def _write_fig(self, metric: BaseMetric, fig: go.Figure, fig_idx: Optional[int] = None) -> None:
        json_fig = fig.to_json()

        basename = f"{metric.name}.json"
        local_path = f"{self.tmp_dir}/{basename}"

        if fig_idx is not None:
            fig_idx = "{:02d}".format(fig_idx)
            basename = f"{metric.name}_{fig_idx}.json"
            local_path = f"{self.tmp_dir}/{basename}"

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(json_fig)

        sly.logger.info("Saved: %r", basename)

    def __enter__(self):
        mkdir(self.tmp_dir, remove_content_if_exists=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if dir_exists(self.tmp_dir):
            remove_dir(self.tmp_dir)
