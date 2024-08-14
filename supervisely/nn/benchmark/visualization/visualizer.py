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
from supervisely.app.widgets import GridGalleryV2
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
    # # WhatIs,
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
    <sly-iw-sidebar :options="{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false,  disableResize: true, sidebarWidth: 300 }">
        <div slot="sidebar">"""

    for vis in metric_visualizations:
        template_str += vis.template_sidebar_str

    template_str += """\n        </div>
      
        <div style="padding: 0 15px;">"""

    for vis in metric_visualizations:
        template_str += vis.template_main_str

    template_str += "\n        </div>\n    </sly-iw-sidebar>"

    template_str += """\n
        <sly-iw-gallery
            ref='modal_general'
            iw-widget-id='modal_general'
            :options="{'isModalWindow': true}"
            :actions="{
                'init': {
                'dataSource': '/data/modal_general.json',
                },
            }"
            :command="command"
            :data="data"
        /> \n</div>"""

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

        self.gt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.gt_project_info.id)
        )
        self.dt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.dt_project_info.id)
        )

        if benchmark.cv_task == CVTask.OBJECT_DETECTION:
            self._initialize_object_detection_loader()
        else:
            raise NotImplementedError("Please specify a new CVTask")

    def _initialize_object_detection_loader(self):

        cocoGt_path, cocoDt_path, eval_data_path, inference_info_path = (
            self.eval_dir + "/cocoGt.json",
            self.eval_dir + "/cocoDt.json",
            self.eval_dir + "/eval_data.pkl",
            self.eval_dir + "/inference_info.json",
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

        inference_info = {}
        if file_exists(inference_info_path):
            with open(inference_info_path, "r") as f:
                inference_info = json.load(f)
            self.inference_info = inference_info
        else:
            self.inference_info = self._benchmark._eval_inference_info

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
            self.mp.confidence_score_profile(), columns=["scores", "precision", "recall", "f1"]
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

        self._update_pred_dcts()
        self._update_diff_dcts()

    def visualize(self):
        mkdir(f"{self.layout_dir}/data", remove_content_if_exists=True)

        initialized = [mv(self) for mv in _METRIC_VISUALIZATIONS]
        initialized = [mv for mv in initialized if self.cv_task.value in mv.cv_tasks]
        for mv in initialized:
            for widget in mv.schema:
                self._write_markdown_files(mv, widget)
                self._write_json_files(mv, widget)

        res = {}
        gallery = GridGalleryV2(
            columns_number=3,
            enable_zoom=False,
            default_tag_filters=[{"confidence": [0.6, 1]}, {"outcome": "TP"}],
            show_zoom_slider=False,
        )
        gallery._update_filters()
        res.update(gallery.get_json_state())

        # res.update(gallery.get_json_data()["content"])
        # res["layoutData"] = res.pop("annotations")

        # res["projectMeta"] = self.dt_project_meta.to_json()
        self.dt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.dt_project_info.id)
        )
        res["projectMeta"] = self.dt_project_meta.to_json()
        basename = "modal_general.json"
        local_path = f"{self.layout_dir}/data/{basename}"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(ujson.dumps(res))
        logger.info("Saved: %r", basename)

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
                    basename = f"{widget.name}_{mv.name}_click_data.json"
                    local_path = f"{self.layout_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(ujson.dumps(click_data))
                    logger.info("Saved: %r", basename)

                    # modal_data = mv.get_modal_data(widget)
                    # basename = f"{widget.name}_{mv.name}_modal_data.json"
                    # local_path = f"{self.layout_dir}/data/{basename}"
                    # with open(local_path, "w", encoding="utf-8") as f:
                    #     f.write(ujson.dumps(modal_data))
                    # logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Gallery):
            content = mv.get_gallery(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

                click_data = mv.get_gallery_click_data(widget)
                if click_data is not None:
                    basename = f"{widget.name}_{mv.name}_click_data.json"
                    local_path = f"{self.layout_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(ujson.dumps(click_data))
                    logger.info("Saved: %r", basename)

                    # modal_data = mv.get_gallery_modal(widget)
                    # basename = f"{widget.name}_{mv.name}_modal_data.json"
                    # local_path = f"{self.layout_dir}/data/{basename}"
                    # with open(local_path, "w", encoding="utf-8") as f:
                    #     f.write(ujson.dumps(modal_data))
                    # logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Table):
            content = mv.get_table(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(content))
                logger.info("Saved: %r", basename)

                content = mv.get_table_click_data(widget)
                basename = f"{widget.name}_{mv.name}_click_data.json"
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

    def update_diff_annotations(self):
        self._add_tags_to_dt_project(self.mp.matches, self.dt_project_info.id)
        gt_project_path, dt_project_path = self._benchmark._download_projects()

        gt_project = Project(gt_project_path, OpenMode.READ)
        dt_project = Project(dt_project_path, OpenMode.READ)
        dt_project_meta = self.dt_project_meta

        dt_images_dct = {}
        dt_anns_dct = {}
        names_dct = {}

        for dataset in dt_project.datasets:
            dataset: Dataset
            paths = list_files(dt_project_path + f"/{dataset.name}/img_info")
            infos = [ImageInfo(**json.load(open(path, "r"))) for path in paths]
            image_names = [x.name for x in sorted(infos, key=lambda info: info.id)]

            dt_anns_dct[dataset.name] = [
                dataset.get_ann(name, dt_project_meta) for name in image_names
            ]
            dt_images_dct[dataset.name] = [dataset.get_image_info(name) for name in image_names]
            names_dct[dataset.name] = image_names

        gt_anns_dct = {}
        for dataset in gt_project.datasets:
            gt_anns_dct[dataset.name] = [
                dataset.get_ann(name, gt_project.meta) for name in names_dct[dataset.name]
            ]

        # matched_ids = []
        # for dataset in dt_project.datasets:
        #     for dt_ann in dt_anns_dct[dataset.name]:
        #         for label in dt_ann.labels:
        #             matched_gt_id = label.tags.get("matched_gt_id")
        #             if matched_gt_id is not None:
        #                 matched_ids.append(matched_gt_id.value)

        matched_id_map = self._get_matched_id_map()  # dt_id -> gt_id
        matched_gt_ids = set(matched_id_map.values())

        new_tag = TagMeta(
            "outcome",
            value_type=TagValueType.ONEOF_STRING,
            possible_values=["TP", "FP", "FN"],
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )
        tag_metas = dt_project_meta.tag_metas
        if dt_project_meta.get_tag_meta(new_tag.name) is None:
            tag_metas = dt_project_meta.tag_metas.add(new_tag)

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
                            # match_tag_id = label.tags.get("matched_gt_id")
                            match_tag_id = matched_id_map.get(label.geometry.sly_id)

                            if match_tag_id is not None:
                                new = label.clone(tags=label.tags.add(Tag(new_tag, "TP")))
                            else:
                                new = label.clone(tags=label.tags.add(Tag(new_tag, "FP")))

                            labels.append(new)

                        dt_anns_new.append(Annotation(gt_ann.img_size, labels))

                        for label in gt_ann.labels:
                            if label.geometry.sly_id not in matched_gt_ids and isinstance(
                                label.geometry, Rectangle
                            ):
                                conf_meta = dt_project_meta.get_tag_meta("confidence")

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

                diff_images = self._api.image.copy_batch(dataset.id, dt_image_ids)

                diff_image_ids = [image.id for image in diff_images]
                self._api.annotation.upload_anns(diff_image_ids, diff_anns_new, progress_cb=pbar1)

        self._update_pred_dcts()
        self._update_diff_dcts()

    def _update_pred_dcts(self):
        datasets = self._api.dataset.get_list(self.gt_project_info.id)
        tmp = {}
        self.gt_images_dct = {}
        self.gt_images_dct_by_name = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            tmp[d.id] = [x.id for x in images]
            for info in images:
                self.gt_images_dct[info.id] = info
                self.gt_images_dct_by_name[info.name] = info

    def _update_diff_dcts(self):
        datasets = self._api.dataset.get_list(self.diff_project_info.id)
        tmp = {}
        self.diff_images_dct = {}
        self.diff_images_dct_by_name = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            tmp[d.id] = [x.id for x in images]
            for info in images:
                self.diff_images_dct[info.id] = info
                self.diff_images_dct_by_name[info.name] = info

    def _add_tags_to_dt_project(self, matches: list, dt_project_id: int):
        api = self._api
        match_tag_meta = TagMeta(
            "matched_gt_id", TagValueType.ANY_NUMBER, applicable_to=TagApplicableTo.OBJECTS_ONLY
        )
        iou_tag_meta = TagMeta(
            "iou", TagValueType.ANY_NUMBER, applicable_to=TagApplicableTo.OBJECTS_ONLY
        )

        # update project meta with new tag metas
        meta = api.project.get_meta(dt_project_id)
        meta = ProjectMeta.from_json(meta)
        meta_old = meta
        if not meta.tag_metas.has_key("matched_gt_id"):
            meta = meta.add_tag_meta(match_tag_meta)
        if not meta.tag_metas.has_key("iou"):
            meta = meta.add_tag_meta(iou_tag_meta)
        if meta != meta_old:
            meta = api.project.update_meta(dt_project_id, meta)
            self.dt_project_meta = meta

        # get tag metas
        # outcome_tag_meta = meta.get_tag_meta("outcome")
        match_tag_meta = meta.get_tag_meta("matched_gt_id")
        iou_tag_meta = meta.get_tag_meta("iou")

        # mappings
        gt_ann_mapping = self.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.click_data.dt_id_mapper.map_obj

        # add tags to objects
        logger.info("Adding tags to DT project")
        with tqdm_sly(desc="Adding tags to DT project", total=len(matches)) as pbar:
            for match in matches:
                if match["type"] == "TP":
                    outcome = "TP"
                    matched_gt_id = gt_ann_mapping[match["gt_id"]]
                    ann_dt_id = dt_ann_mapping[match["dt_id"]]
                    iou = match["iou"]
                    # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
                    if matched_gt_id is not None:
                        api.advanced.add_tag_to_object(
                            match_tag_meta.sly_id, ann_dt_id, int(matched_gt_id)
                        )
                        api.advanced.add_tag_to_object(iou_tag_meta.sly_id, ann_dt_id, float(iou))
                    else:
                        continue
                elif match["type"] == "FP":
                    outcome = "FP"
                    # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
                elif match["type"] == "FN":
                    outcome = "FN"
                else:
                    raise ValueError(f"Unknown match type: {match['type']}")

                pbar.update(1)

    def _get_matched_id_map(self):
        gt_ann_mapping = self.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.click_data.dt_id_mapper.map_obj
        dtId2matched_gt_id = {}
        for match in self.mp.matches:
            if match["type"] == "TP":
                dtId2matched_gt_id[dt_ann_mapping[match["dt_id"]]] = gt_ann_mapping[match["gt_id"]]
        return dtId2matched_gt_id
