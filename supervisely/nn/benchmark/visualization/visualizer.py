from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Tuple

import pandas as pd
from jinja2 import Template

from supervisely._utils import batched
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import file_exists, mkdir
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta

if TYPE_CHECKING:
    from supervisely.nn.benchmark.base_benchmark import BaseBenchmark

from supervisely.nn.benchmark.evaluation.coco.metric_provider import (
    MetricProvider,
)
from supervisely.nn.benchmark.visualization.vis_click_data import ClickData, IdMapper
from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_metrics import ALL_METRICS
from supervisely.nn.benchmark.visualization.vis_templates import generate_main_template
from supervisely.nn.benchmark.visualization.vis_widgets import Widget
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


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
        self._docs_link = "https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/"

        if benchmark.cv_task == CVTask.OBJECT_DETECTION:
            self._initialize_object_detection_loader()
            self.docs_link = self._docs_link + CVTask.OBJECT_DETECTION.value.replace("_", "-")
        else:
            raise NotImplementedError(f"CV task {benchmark.cv_task} is not supported yet")

        self.pbar = benchmark.pbar

    def _initialize_object_detection_loader(self):
        from pycocotools.coco import COCO  # pylint: disable=import-error

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

        self.mp = MetricProvider(
            eval_data["matches"],
            eval_data["coco_metrics"],
            eval_data["params"],
            cocoGt,
            cocoDt,
        )
        self.mp.calculate()

        self.df_score_profile = pd.DataFrame(
            self.mp.confidence_score_profile(), columns=["scores", "precision", "recall", "f1"]
        )

        # downsample
        if len(self.df_score_profile) > 5000:
            self.dfsp_down = self.df_score_profile.iloc[:: len(self.df_score_profile) // 1000]
        else:
            self.dfsp_down = self.df_score_profile

        self.f1_optimal_conf = self.mp.get_f1_optimal_conf()[0]
        if self.f1_optimal_conf is None:
            self.f1_optimal_conf = 0.01
            logger.warn("F1 optimal confidence cannot be calculated. Using 0.01 as default.")

        # Click data
        gt_id_mapper = IdMapper(cocoGt_dataset)
        dt_id_mapper = IdMapper(cocoDt_dataset)

        self.click_data = ClickData(self.mp.m, gt_id_mapper, dt_id_mapper)
        self.base_metrics = self.mp.base_metrics

        self._update_pred_dcts()
        self._update_gt_dcts()
        self._update_diff_dcts()

        self._objects_bindings = []

    def visualize(self):
        from supervisely.app.widgets import GridGalleryV2

        mkdir(f"{self.layout_dir}/data", remove_content_if_exists=True)

        initialized = [mv(self) for mv in ALL_METRICS]
        initialized = [mv for mv in initialized if self.cv_task.value in mv.cv_tasks]
        with self.pbar(
            message="Saving visualization files",
            total=len([w for mv in initialized for w in mv.schema]),
        ) as p:
            for mv in initialized:
                for widget in mv.schema:
                    self._write_markdown_files(mv, widget)
                    self._write_json_files(mv, widget)
                    p.update(1)

        res = {}
        optimal_conf = round(self.f1_optimal_conf, 1)
        gallery = GridGalleryV2(
            columns_number=3,
            enable_zoom=False,
            annotations_opacity=0.4,
            border_width=4,
            default_tag_filters=[{"confidence": [optimal_conf, 1]}],
            show_zoom_slider=False,
        )
        gallery._update_filters()
        res.update(gallery.get_json_state())

        self.dt_project_meta = ProjectMeta.from_json(
            data=self._api.project.get_meta(id=self.dt_project_info.id)
        )
        res["projectMeta"] = self.dt_project_meta.to_json()
        for basename in ["modal_general.json", "modal_general_diff.json"]:
            local_path = f"{self.layout_dir}/data/{basename}"
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(res))
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
                        f.write(json.dumps(click_data))
                    logger.info("Saved: %r", basename)

                    # modal_data = mv.get_modal_data(widget)
                    # basename = f"{widget.name}_{mv.name}_modal_data.json"
                    # local_path = f"{self.layout_dir}/data/{basename}"
                    # with open(local_path, "w", encoding="utf-8") as f:
                    #     f.write(json.dumps(modal_data))
                    # logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Gallery):
            content = mv.get_gallery(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(content))
                logger.info("Saved: %r", basename)

                click_data = mv.get_gallery_click_data(widget)
                if click_data is not None:
                    basename = f"{widget.name}_{mv.name}_click_data.json"
                    local_path = f"{self.layout_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(click_data))
                    logger.info("Saved: %r", basename)

                diff_data = mv.get_diff_gallery_data(widget)
                if diff_data is not None:
                    basename = f"{widget.name}_{mv.name}_diff_data.json"
                    local_path = f"{self.layout_dir}/data/{basename}"
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(diff_data))
                    logger.info("Saved: %r", basename)

                    # modal_data = mv.get_gallery_modal(widget)
                    # basename = f"{widget.name}_{mv.name}_modal_data.json"
                    # local_path = f"{self.layout_dir}/data/{basename}"
                    # with open(local_path, "w", encoding="utf-8") as f:
                    #     f.write(json.dumps(modal_data))
                    # logger.info("Saved: %r", basename)

        if isinstance(widget, Widget.Table):
            content = mv.get_table(widget)
            if content is not None:
                basename = f"{widget.name}_{mv.name}.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(content))
                logger.info("Saved: %r", basename)

                content = mv.get_table_click_data(widget)
                basename = f"{widget.name}_{mv.name}_click_data.json"
                local_path = f"{self.layout_dir}/data/{basename}"
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(content))
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
        meta = self._update_pred_meta_with_tags(self.dt_project_info.id, self.dt_project_meta)
        self._api.project.update_meta(self.diff_project_info.id, meta)

        self.dt_project_meta = meta
        self._add_tags_to_pred_project(self.mp.matches, self.dt_project_info.id)
        gt_project_path, pred_project_path = self._benchmark._download_projects()

        gt_project = Project(gt_project_path, OpenMode.READ)
        pred_project = Project(pred_project_path, OpenMode.READ)

        names_map = {}
        pred_images_map = {}
        pred_anns_map = {}
        total = 0

        for dataset in pred_project.datasets:
            dataset: Dataset
            infos = [dataset.get_item_info(name) for name in dataset.get_items_names()]
            infos = sorted(infos, key=lambda info: info.id)
            image_names = [x.name for x in sorted(infos, key=lambda info: info.id)]
            total += len(image_names)

            pred_images_map[dataset.name] = infos
            names_map[dataset.name] = image_names
            pred_anns_map[dataset.name] = [dataset.get_ann(name, meta) for name in image_names]

        gt_anns_map = {}
        for d in gt_project.datasets:
            gt_anns_map[d.name] = [d.get_ann(name, gt_project.meta) for name in names_map[d.name]]

        matched_id_map = self._get_matched_id_map()  # dt_id -> gt_id
        matched_gt_ids = set(matched_id_map.values())

        outcome_tag = meta.get_tag_meta("outcome")
        conf_meta = meta.get_tag_meta("confidence")
        match_tag = meta.get_tag_meta("matched_gt_id")

        pred_tag_list = []
        with self.pbar(message="Creating diff_project", total=total) as p:
            for dataset in self._api.dataset.get_list(self.diff_project_info.id):
                diff_anns_new = []

                for gt_ann, dt_ann in zip(gt_anns_map[dataset.name], pred_anns_map[dataset.name]):
                    labels = []
                    for label in dt_ann.labels:
                        # match_tag_id = label.tags.get("matched_gt_id")
                        match_tag_id = matched_id_map.get(label.geometry.sly_id)

                        value = "TP" if match_tag_id else "FP"
                        pred_tag_list.append(
                            {
                                "tagId": outcome_tag.sly_id,
                                "figureId": label.geometry.sly_id,
                                "value": value,
                            }
                        )
                        conf = label.tags.get("confidence").value
                        if conf < self.f1_optimal_conf:
                            continue  # do not add labels with low confidence to diff project
                        if match_tag_id:
                            continue  # do not add TP labels to diff project
                        label = label.add_tag(Tag(outcome_tag, value))
                        if not match_tag_id:
                            label = label.add_tag(Tag(match_tag, int(label.geometry.sly_id)))
                        labels.append(label)

                    for label in gt_ann.labels:
                        if label.geometry.sly_id not in matched_gt_ids:
                            if self._is_label_compatible_to_cv_task(label):
                                new_label = label.add_tags(
                                    [Tag(outcome_tag, "FN"), Tag(conf_meta, 1)]
                                )
                                labels.append(new_label)

                    diff_anns_new.append(Annotation(gt_ann.img_size, labels))

                pred_img_ids = [x.id for x in pred_images_map[dataset.name]]
                # self._api.annotation.upload_anns(pred_img_ids, dt_anns_new, progress_cb=pbar2)

                diff_images = self._api.image.copy_batch(dataset.id, pred_img_ids)

                diff_img_ids = [image.id for image in diff_images]
                self._api.annotation.upload_anns(diff_img_ids, diff_anns_new, progress_cb=p.update)

        self._api.image.tag.add_to_objects(self.dt_project_info.id, pred_tag_list)

        self._update_gt_dcts()
        self._update_diff_dcts()
        self._update_pred_dcts()

    def _update_gt_dcts(self):
        datasets = self._api.dataset.get_list(self.gt_project_info.id)
        self.gt_images_dct = {}
        self.gt_images_dct_by_name = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            for info in images:
                self.gt_images_dct[info.id] = info
                self.gt_images_dct_by_name[info.name] = info

    def _update_diff_dcts(self):
        datasets = self._api.dataset.get_list(self.diff_project_info.id)
        self.diff_images_dct = {}
        self.diff_images_dct_by_name = {}
        self.diff_ann_jsons = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            for info in images:
                self.diff_images_dct[info.id] = info
                self.diff_images_dct_by_name[info.name] = info

            diff_anns = self._api.annotation.download_batch(d.id, [x.id for x in images])
            self.diff_ann_jsons.update({ann.image_id: ann.annotation for ann in diff_anns})

    def _update_pred_dcts(self):
        datasets = self._api.dataset.get_list(self.dt_project_info.id)
        self.dt_images_dct = {}
        self.dt_images_dct_by_name = {}
        self.dt_ann_jsons = {}
        for d in datasets:
            images = self._api.image.get_list(d.id)
            for info in images:
                self.dt_images_dct[info.id] = info
                self.dt_images_dct_by_name[info.name] = info

            dt_anns = self._api.annotation.download_batch(d.id, [x.id for x in images])
            self.dt_ann_jsons.update({ann.image_id: ann.annotation for ann in dt_anns})

    def _update_pred_meta_with_tags(self, project_id: int, meta: ProjectMeta) -> ProjectMeta:
        old_meta = meta
        outcome_tag = TagMeta(
            "outcome",
            value_type=TagValueType.ONEOF_STRING,
            possible_values=["TP", "FP", "FN"],
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )
        match_tag = TagMeta(
            "matched_gt_id",
            TagValueType.ANY_NUMBER,
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )
        iou_tag = TagMeta(
            "iou",
            TagValueType.ANY_NUMBER,
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )

        for tag in [outcome_tag, match_tag, iou_tag]:
            if meta.get_tag_meta(tag.name) is None:
                meta = meta.add_tag_meta(tag)

        if old_meta == meta:
            return meta

        meta = self._api.project.update_meta(project_id, meta)
        return meta

        # conf_meta = meta.get_tag_meta("confidence")

    def _add_tags_to_pred_project(self, matches: list, pred_project_id: int):

        # get tag metas
        # outcome_tag_meta = self.dt_project_meta.get_tag_meta("outcome")
        match_tag_meta = self.dt_project_meta.get_tag_meta("matched_gt_id")
        iou_tag_meta = self.dt_project_meta.get_tag_meta("iou")

        # mappings
        gt_ann_mapping = self.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.click_data.dt_id_mapper.map_obj

        # add tags to objects
        logger.info("Adding tags to DT project")

        with self.pbar(message="Adding tags to DT project", total=len(matches)) as p:
            for batch in batched(matches, 100):
                pred_tag_list = []
                for match in batch:
                    if match["type"] == "TP":
                        outcome = "TP"
                        matched_gt_id = gt_ann_mapping[match["gt_id"]]
                        ann_dt_id = dt_ann_mapping[match["dt_id"]]
                        iou = match["iou"]
                        # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
                        if matched_gt_id is not None:
                            pred_tag_list.extend(
                                [
                                    {
                                        "tagId": match_tag_meta.sly_id,
                                        "figureId": ann_dt_id,
                                        "value": int(matched_gt_id),
                                    },
                                    {
                                        "tagId": iou_tag_meta.sly_id,
                                        "figureId": ann_dt_id,
                                        "value": float(iou),
                                    },
                                ]
                            )
                        else:
                            continue
                    elif match["type"] == "FP":
                        outcome = "FP"
                        # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
                    elif match["type"] == "FN":
                        outcome = "FN"
                    else:
                        raise ValueError(f"Unknown match type: {match['type']}")

                self._api.image.tag.add_to_objects(pred_project_id, pred_tag_list)
                p.update(len(batch))

    def _get_matched_id_map(self):
        gt_ann_mapping = self.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.click_data.dt_id_mapper.map_obj
        dtId2matched_gt_id = {}
        for match in self.mp.matches_filtered:
            if match["type"] == "TP":
                dtId2matched_gt_id[dt_ann_mapping[match["dt_id"]]] = gt_ann_mapping[match["gt_id"]]
        return dtId2matched_gt_id

    def _is_label_compatible_to_cv_task(self, label):
        if self.cv_task == CVTask.OBJECT_DETECTION:
            return isinstance(label.geometry, Rectangle)
        return False
