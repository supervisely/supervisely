from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import pandas as pd
import ujson
from jinja2 import Template
from pycocotools.coco import COCO

from supervisely._utils import *
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.api import Api
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import *
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta

if TYPE_CHECKING:
    from supervisely.nn.benchmark.base_benchmark import BaseBenchmark

from supervisely.nn.benchmark.evaluation.object_detection.metric_provider import (
    MetricProvider,
)
from supervisely.nn.benchmark.visualization.visualizations import *
from supervisely.nn.benchmark.visualization.visualizations import MetricVis
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

_METRIC_VISUALIZATIONS = (
    Overview,
    ExplorerGrid,
    ModelPredictions,
    WhatIs,
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
    F1ScoreAtDifferentIOU,
    ConfidenceDistribution,
    PerClassAvgPrecision,
    PerClassOutcomeCounts,
    # segmentation-only
    # # TODO integrate binary files while saving to self.layout_dir to the current solution
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


class Visualizer:

    def __init__(self, benchmark: BaseBenchmark) -> None:

        self._benchmark = benchmark
        self._api = benchmark.api
        self.cv_task = benchmark.cv_task

        self.eval_dir = benchmark.get_eval_results_dir()
        self.layout_dir = benchmark.get_layout_results_dir()

        self.dt_project_info = benchmark.dt_project_info
        self.gt_project_info = benchmark.gt_project_info
        self.diff_project_info = benchmark.diff_project_info

        if benchmark.cv_task == CVTask.OBJECT_DETECTION:
            self._initialize_object_detection_loader()
        else:
            raise NotImplementedError("Please specify a new CVTask")

    def _initialize_object_detection_loader(self):

        cocoGt_path, cocoDt_path, eval_data_path = (
            self.eval_dir + "/cocoGt.json",
            self.eval_dir + "/cocoDt.json",
            self.eval_dir + "/eval_data.pkl",
        )

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

        mp = MetricProvider(
            eval_data["matches"],
            eval_data["coco_metrics"],
            eval_data["params"],
            cocoGt,
            cocoDt,
        )
        mp.calculate()
        self.mp = mp

        self.df_score_profile = pd.DataFrame(
            self.mp.confidence_score_profile(), columns=["scores", "Precision", "Recall", "F1"]
        )

        # downsample
        if len(self.df_score_profile) > 5000:
            self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        else:
            self.dfsp_down = self.df_score_profile

        # Click data
        gt_id_mapper = IdMapper(cocoGt_dataset)
        dt_id_mapper = IdMapper(cocoDt_dataset)

        self.click_data = ClickData(self.mp.m, gt_id_mapper, dt_id_mapper)

        self.base_metrics = self.mp.base_metrics

        self.dt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.dt_project_info.id)
        )
        datasets = self._api.dataset.get_list(self.dt_project_info.id)

        tmp = {}
        self.dt_images_dct = {}
        self.dt_images_dct_by_name = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            tmp[d.id] = [x.id for x in images]
            for info in images:
                self.dt_images_dct[info.id] = info
                self.dt_images_dct_by_name[info.name] = info

        self.dt_ann_jsons = {
            ann.image_id: ann.annotation
            for d in datasets
            for ann in self._api.annotation.download_batch(d.id, tmp[d.id])
        }

    def visualize(self):
        mkdir(f"{self.layout_dir}/data", remove_content_if_exists=True)

        initialized = [mv(self) for mv in _METRIC_VISUALIZATIONS]
        initialized = [mv for mv in initialized if self.cv_task.value in mv.cv_tasks]
        for mv in initialized:
            for widget in mv.schema:
                self._write_markdown_files(mv, widget)
                self._write_json_files(mv, widget)
        self._save_template(initialized)

    def _write_markdown_files(self, metric_visualization: MetricVis, widget: Widget):

        if isinstance(widget, Widget.Markdown):
            content = metric_visualization.get_md_content(widget)
            local_path = f"{self.layout_dir}/data/{widget.name}.md"
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info("Saved: %r", f"{widget.name}.md")

        if isinstance(widget, Widget.Collapse):
            for subwidget in widget.schema:
                content = metric_visualization.get_md_content(subwidget)
                local_path = f"{self.layout_dir}/data/{subwidget.name}.md"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)

                logger.info("Saved: %r", f"{subwidget.name}.md")

    def _write_json_files(self, mv: MetricVis, widget: Widget):

        if isinstance(widget, Widget.Chart):
            fig = mv.get_figure(widget)
            if fig is not None:
                fig_data = {
                    "selected": None,
                    "galleryContent": "",
                    "dialogVisible": False,
                    "chartContent": json.loads(fig.to_json()),
                }
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(fig_data, f)
                logger.info("Saved: %r", basename)

                click_data = mv.get_click_data(widget)
                if click_data is not None:
                    basename = f"{widget.name}_{mv.name}_clickdata.json"
                    local_path = f"{self.layout_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(ujson.dumps(click_data))
                    logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Gallery):
            content = mv.get_gallery(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Table):
            content = mv.get_table(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

    def _generate_template(self, metric_visualizations: Tuple[MetricVis]) -> str:
        html_snippets = {}
        main_template = Template(generate_main_template(metric_visualizations))
        for mv in metric_visualizations:
            for widget in mv.schema:
                if isinstance(widget, Widget.Notification):
                    html_snippets.update(mv.get_html_snippets())

            html_snippets.update(mv.get_html_snippets())

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
        local_path = f"{self.layout_dir}/template.vue"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(self._generate_template(metric_visualizations))
        logger.info("Saved: %r", "template.vue")
        local_path = f"{self.layout_dir}/state.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(self._generate_state(metric_visualizations), f)
        logger.info("Saved: %r", "state.json")

    def update_annotations(self):
        gt_project_path, dt_project_path = self._benchmark._download_projects()

        gt_project = Project(gt_project_path, OpenMode.READ)
        dt_project = Project(dt_project_path, OpenMode.READ)

        dt_images_dct = {}
        dt_anns_dct = {}
        names_dct = {}
        for dataset in dt_project.datasets:
            dataset: Dataset
            paths = list_files(dt_project_path + f"/{dataset.name}/img_info")
            image_names = [os.path.basename(x) for x in paths]
            image_names = [".".join(x.split(".")[:-1]) for x in image_names]

            dt_anns_dct[dataset.name] = [
                dataset.get_ann(name, dt_project.meta) for name in image_names
            ]
            dt_images_dct[dataset.name] = [dataset.get_image_info(name) for name in image_names]
            names_dct[dataset.name] = image_names

        gt_anns_dct = {}
        for dataset in gt_project.datasets:
            gt_anns_dct[dataset.name] = [
                dataset.get_ann(name, gt_project.meta) for name in names_dct[dataset.name]
            ]

        matched_ids = []
        for dataset in dt_project.datasets:
            for dt_ann in dt_anns_dct[dataset.name]:
                for label in dt_ann.labels:
                    matched_gt_id = label.tags.get("matched_gt_id")
                    if matched_gt_id is not None:
                        matched_ids.append(matched_gt_id.value)

        new_tag = TagMeta(
            "outcome",
            value_type=TagValueType.ONEOF_STRING,
            possible_values=["TP", "FP", "FN"],
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )
        tag_metas = dt_project.meta.tag_metas
        if dt_project.meta.get_tag_meta(new_tag.name) is None:
            tag_metas = dt_project.meta.tag_metas.add(new_tag)

        diff_meta = ProjectMeta(
            obj_classes=gt_project.meta.obj_classes,
            tag_metas=tag_metas,
        )

        self._api.project.update_meta(self.diff_project_info.id, diff_meta.to_json())
        self._api.project.update_meta(self.dt_project_info.id, diff_meta.to_json())

        with tqdm_sly(
            desc="Creating diff_project", total=sum([len(x) for x in gt_anns_dct.values()])
        ) as pbar1:
            with tqdm_sly(
                desc="Updating dt_project", total=sum([len(x) for x in gt_anns_dct.values()])
            ) as pbar2:

                for dataset in self._api.dataset.get_list(self.diff_project_info.id):
                    diff_anns_new, dt_anns_new = [], []

                    for gt_ann, dt_ann in zip(gt_anns_dct[dataset.name], dt_anns_dct[dataset.name]):
                        labels = []
                        for label in dt_ann.labels:
                            match_tag_id = label.tags.get("matched_gt_id")
                            if match_tag_id is not None:
                                new = label.clone(tags=label.tags.add(Tag(new_tag, "TP")))
                            else:
                                new = label.clone(tags=label.tags.add(Tag(new_tag, "FP")))

                            labels.append(new)

                        dt_anns_new.append(Annotation(gt_ann.img_size, labels))

                        for label in gt_ann.labels:
                            if label.geometry.sly_id not in matched_ids and isinstance(
                                label.geometry, Rectangle
                            ):
                                conf_meta = dt_project.meta.get_tag_meta("confidence")
                                labels.append(
                                    label.clone(
                                        tags=label.tags.add_items(
                                            [Tag(new_tag, "FN"), Tag(conf_meta, 1)]
                                        )
                                    )
                                )

                        diff_anns_new.append(Annotation(gt_ann.img_size, labels))

                    dt_image_ids = [x.id for x in dt_images_dct[dataset.name]]
                    self._api.annotation.upload_anns(dt_image_ids, dt_anns_new, progress_cb=pbar2)

                    try:
                        diff_images = self._api.image.copy_batch(dataset.id, dt_image_ids)
                    except ValueError:
                        diff_images = self._api.image.get_list(dataset.id)
                    diff_image_ids = [image.id for image in diff_images]
                    self._api.annotation.upload_anns(
                        diff_image_ids, diff_anns_new, progress_cb=pbar1
                    )

        pass
