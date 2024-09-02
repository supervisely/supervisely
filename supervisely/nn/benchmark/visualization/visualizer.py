from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Dict, List, Tuple

import pandas as pd
from jinja2 import Template

from supervisely import AnyGeometry, Bitmap, Polygon, Rectangle
from supervisely._utils import batched
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.image_api import ImageInfo
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.io import fs
from supervisely.io.fs import file_exists, mkdir
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta

if TYPE_CHECKING:
    from supervisely.nn.benchmark.base_benchmark import BaseBenchmark

from supervisely import Label
from supervisely.nn.benchmark.evaluation.coco.metric_provider import MetricProvider
from supervisely.nn.benchmark.visualization.vis_click_data import ClickData, IdMapper
from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_metrics import ALL_METRICS
from supervisely.nn.benchmark.visualization.vis_templates import generate_main_template
from supervisely.nn.benchmark.visualization.vis_widgets import Widget
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class ImageComparisonData:
    def __init__(
        self,
        gt_image_info: ImageInfo = None,
        pred_image_info: ImageInfo = None,
        diff_image_info: ImageInfo = None,
        gt_annotation: Annotation = None,
        pred_annotation: Annotation = None,
        diff_annotation: Annotation = None,
    ):
        self.gt_image_info = gt_image_info
        self.pred_image_info = pred_image_info
        self.diff_image_info = diff_image_info
        self.gt_annotation = gt_annotation
        self.pred_annotation = pred_annotation
        self.diff_annotation = diff_annotation


class Visualizer:

    def __init__(self, benchmark: BaseBenchmark) -> None:

        if benchmark.dt_project_info is None:
            raise RuntimeError(
                "The benchmark prediction project was not initialized. Please run evaluation or specify dt_project_info property of benchmark object."
            )

        eval_dir = benchmark.get_eval_results_dir()
        assert not fs.dir_empty(
            eval_dir
        ), f"The result dir {eval_dir!r} is empty. You should run evaluation before visualizing results."

        self._benchmark = benchmark
        self._api = benchmark.api
        self.cv_task = benchmark.cv_task

        self.eval_dir = benchmark.get_eval_results_dir()
        self.layout_dir = benchmark.get_layout_results_dir()

        self.dt_project_info = benchmark.dt_project_info
        self.gt_project_info = benchmark.gt_project_info
        self._benchmark.diff_project_info, existed = self._benchmark._get_or_create_diff_project()
        self.diff_project_info = benchmark.diff_project_info
        self.classes_whitelist = benchmark.classes_whitelist
        self.diff_project_meta = ProjectMeta.from_json(
            self._api.project.get_meta(self.diff_project_info.id)
        )
        self.comparison_data: Dict[int, ImageComparisonData] = {}  # gt_id -> ImageComparisonData

        self.gt_project_meta = self._get_filtered_project_meta(self.gt_project_info.id)
        self.dt_project_meta = self._get_filtered_project_meta(self.dt_project_info.id)
        self._docs_link = "https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/"

        if benchmark.cv_task == CVTask.OBJECT_DETECTION:
            self._initialize_object_detection_loader()
            self.docs_link = self._docs_link + CVTask.OBJECT_DETECTION.value.replace("_", "-")
        elif benchmark.cv_task == CVTask.INSTANCE_SEGMENTATION:
            self._initialize_instance_segmentation_loader()
            self.docs_link = self._docs_link + CVTask.INSTANCE_SEGMENTATION.value.replace("_", "-")
        else:
            raise NotImplementedError(f"CV task {benchmark.cv_task} is not supported yet")

        self.pbar = benchmark.pbar

        if not existed:
            self.update_diff_annotations()
        else:
            self._init_comparison_data()

    def _initialize_object_detection_loader(self):
        from pycocotools.coco import COCO  # pylint: disable=import-error

        from supervisely.nn.benchmark.visualization import vis_texts

        self.vis_texts = vis_texts

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

        self._objects_bindings = []

    def _initialize_instance_segmentation_loader(self):
        from pycocotools.coco import COCO  # pylint: disable=import-error

        from supervisely.nn.benchmark.visualization.instance_segmentation import (
            text_template,
        )

        self.vis_texts = text_template

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
        gallery = GridGalleryV2(
            columns_number=3,
            enable_zoom=False,
            annotations_opacity=0.4,
            border_width=4,
            default_tag_filters=[{"confidence": [self.f1_optimal_conf, 1]}],
            show_zoom_slider=False,
        )
        gallery._update_filters()
        res.update(gallery.get_json_state())

        self.dt_project_meta = self._get_filtered_project_meta(self.dt_project_info.id)
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
        self._update_diff_meta(meta)

        self.dt_project_meta = meta
        self._add_tags_to_pred_project(self.mp.matches, self.dt_project_info.id)
        gt_project_path, pred_project_path = self._benchmark._download_projects(save_images=False)

        gt_project = Project(gt_project_path, OpenMode.READ)
        pred_project = Project(pred_project_path, OpenMode.READ)
        diff_dataset_name_to_info = {
            ds.name: ds for ds in self._api.dataset.get_list(self.diff_project_info.id)
        }

        matched_id_map = self._get_matched_id_map()  # dt_id -> gt_id
        matched_gt_ids = set(matched_id_map.values())

        outcome_tag = meta.get_tag_meta("outcome")
        conf_meta = meta.get_tag_meta("confidence")
        if conf_meta is None:
            conf_meta = meta.get_tag_meta("conf")
        match_tag = meta.get_tag_meta("matched_gt_id")

        pred_tag_list = []
        with self.pbar(message="Creating diff_project", total=pred_project.total_items) as progress:
            for pred_dataset in pred_project.datasets:
                pred_dataset: Dataset
                gt_dataset: Dataset = gt_project.datasets.get(pred_dataset.name)
                diff_dataset_info = diff_dataset_name_to_info[pred_dataset.name]
                diff_anns = []
                gt_image_ids = []
                pred_img_ids = []
                for item_name in pred_dataset.get_items_names():
                    gt_image_info = gt_dataset.get_image_info(item_name)
                    gt_image_ids.append(gt_image_info.id)
                    pred_image_info = pred_dataset.get_image_info(item_name)
                    pred_img_ids.append(pred_image_info.id)
                    gt_ann = gt_dataset.get_ann(item_name, gt_project.meta)
                    pred_ann = pred_dataset.get_ann(item_name, pred_project.meta)
                    labels = []

                    # TP and FP
                    for label in pred_ann.labels:
                        match_tag_id = matched_id_map.get(label.geometry.sly_id)
                        value = "TP" if match_tag_id else "FP"
                        pred_tag_list.append(
                            {
                                "tagId": outcome_tag.sly_id,
                                "figureId": label.geometry.sly_id,
                                "value": value,
                            }
                        )
                        conf = 1
                        for tag in label.tags.items():
                            tag: Tag
                            if tag.name in ["confidence", "conf"]:
                                conf = tag.value
                                break

                        if conf < self.f1_optimal_conf:
                            continue  # do not add labels with low confidence to diff project
                        if match_tag_id:
                            continue  # do not add TP labels to diff project
                        label = label.add_tag(Tag(outcome_tag, value))
                        label = label.add_tag(Tag(match_tag, int(label.geometry.sly_id)))
                        labels.append(label)

                    # FN
                    for label in gt_ann.labels:
                        if self.classes_whitelist:
                            if label.obj_class.name not in self.classes_whitelist:
                                continue
                        if label.geometry.sly_id not in matched_gt_ids:
                            if self._is_label_compatible_to_cv_task(label):
                                new_label = label.add_tags(
                                    [Tag(outcome_tag, "FN"), Tag(conf_meta, 1)]
                                )
                                labels.append(new_label)

                    diff_ann = Annotation(gt_ann.img_size, labels)
                    diff_anns.append(diff_ann)

                    # comparison data
                    self._update_comparison_data(
                        gt_image_info.id,
                        gt_image_info=gt_image_info,
                        pred_image_info=pred_image_info,
                        gt_annotation=gt_ann,
                        pred_annotation=pred_ann,
                        diff_annotation=diff_ann,
                    )

                diff_img_infos = self._api.image.copy_batch(diff_dataset_info.id, pred_img_ids)
                self._api.annotation.upload_anns(
                    [img_info.id for img_info in diff_img_infos],
                    diff_anns,
                    progress_cb=progress.update,
                )
                for gt_img_id, diff_img_info in zip(gt_image_ids, diff_img_infos):
                    self._update_comparison_data(gt_img_id, diff_image_info=diff_img_info)

        self._api.image.tag.add_to_objects(self.dt_project_info.id, pred_tag_list)

    def _init_comparison_data(self):
        gt_project_path, pred_project_path = self._benchmark._download_projects(save_images=False)
        gt_project = Project(gt_project_path, OpenMode.READ)
        pred_project = Project(pred_project_path, OpenMode.READ)
        diff_dataset_name_to_info = {
            ds.name: ds for ds in self._api.dataset.get_list(self.diff_project_info.id)
        }

        for pred_dataset in pred_project.datasets:
            pred_dataset: Dataset
            gt_dataset: Dataset = gt_project.datasets.get(pred_dataset.name)
            try:
                diff_dataset_info = diff_dataset_name_to_info[pred_dataset.name]
            except KeyError:
                raise RuntimeError(
                    f"Difference project was not created properly. Dataset {pred_dataset.name} is missing"
                )

            for item_names_batch in batched(pred_dataset.get_items_names(), 100):
                # diff project may be not created yet
                item_names_batch.sort()
                try:
                    diff_img_infos_batch: List[ImageInfo] = sorted(
                        self._api.image.get_list(
                            diff_dataset_info.id,
                            filters=[
                                {"field": "name", "operator": "in", "value": item_names_batch}
                            ],
                        ),
                        key=lambda x: x.name,
                    )
                    diff_anns_batch_dict = {
                        ann_info.image_id: Annotation.from_json(
                            ann_info.annotation, self.diff_project_meta
                        )
                        for ann_info in self._api.annotation.download_batch(
                            diff_dataset_info.id, [img_info.id for img_info in diff_img_infos_batch]
                        )
                    }
                    assert (
                        len(item_names_batch)
                        == len(diff_img_infos_batch)
                        == len(diff_anns_batch_dict)
                    ), "Some images are missing in the difference project"

                    for item_name, diff_img_info in zip(item_names_batch, diff_img_infos_batch):
                        assert (
                            item_name == diff_img_info.name
                        ), "Image names in difference project and prediction project do not match"
                        gt_image_info = gt_dataset.get_image_info(item_name)
                        pred_image_info = pred_dataset.get_image_info(item_name)
                        gt_ann = gt_dataset.get_ann(item_name, gt_project.meta)
                        pred_ann = pred_dataset.get_ann(item_name, pred_project.meta)
                        diff_ann = diff_anns_batch_dict[diff_img_info.id]

                        self._update_comparison_data(
                            gt_image_info.id,
                            gt_image_info=gt_image_info,
                            pred_image_info=pred_image_info,
                            diff_image_info=diff_img_info,
                            gt_annotation=gt_ann,
                            pred_annotation=pred_ann,
                            diff_annotation=diff_ann,
                        )
                except Exception:
                    raise RuntimeError("Difference project was not created properly")

    def _update_comparison_data(
        self,
        gt_image_id: int,
        gt_image_info: ImageInfo = None,
        pred_image_info: ImageInfo = None,
        diff_image_info: ImageInfo = None,
        gt_annotation: Annotation = None,
        pred_annotation: Annotation = None,
        diff_annotation: Annotation = None,
    ):
        comparison_data = self.comparison_data.get(gt_image_id, None)
        if comparison_data is None:
            self.comparison_data[gt_image_id] = ImageComparisonData(
                gt_image_info=gt_image_info,
                pred_image_info=pred_image_info,
                diff_image_info=diff_image_info,
                gt_annotation=gt_annotation,
                pred_annotation=pred_annotation,
                diff_annotation=diff_annotation,
            )
        else:
            for attr, value in {
                "gt_image_info": gt_image_info,
                "pred_image_info": pred_image_info,
                "diff_image_info": diff_image_info,
                "gt_annotation": gt_annotation,
                "pred_annotation": pred_annotation,
                "diff_annotation": diff_annotation,
            }.items():
                if value is not None:
                    setattr(comparison_data, attr, value)

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
        confidence_tag = TagMeta(
            "confidence",
            value_type=TagValueType.ANY_NUMBER,
            applicable_to=TagApplicableTo.OBJECTS_ONLY,
        )

        for tag in [outcome_tag, match_tag, iou_tag]:
            if meta.get_tag_meta(tag.name) is None:
                meta = meta.add_tag_meta(tag)

        if meta.get_tag_meta("confidence") is None and meta.get_tag_meta("conf") is None:
            meta = meta.add_tag_meta(confidence_tag)

        if old_meta == meta:
            return meta

        meta = self._api.project.update_meta(project_id, meta)
        return meta

    def _update_diff_meta(self, meta: ProjectMeta):
        new_obj_classes = []
        for obj_class in meta.obj_classes:
            new_obj_classes.append(obj_class.clone(geometry_type=AnyGeometry))
        meta = meta.clone(obj_classes=new_obj_classes)
        self.diff_project_meta = self._api.project.update_meta(self.diff_project_info.id, meta)

    def _update_diff_meta(self, meta: ProjectMeta):
        new_obj_classes = []
        for obj_class in meta.obj_classes:
            new_obj_classes.append(obj_class.clone(geometry_type=AnyGeometry))
        meta = meta.clone(obj_classes=new_obj_classes)
        self.diff_project_meta = self._api.project.update_meta(self.diff_project_info.id, meta)

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

    def _is_label_compatible_to_cv_task(self, label: Label):
        if self.cv_task == CVTask.OBJECT_DETECTION:
            return isinstance(label.geometry, Rectangle)
        if self.cv_task == CVTask.INSTANCE_SEGMENTATION:
            return isinstance(label.geometry, (Bitmap, Polygon))
        return False

    def _get_filtered_project_meta(self, project_id: int) -> ProjectMeta:
        meta = self._api.project.get_meta(project_id)
        meta = ProjectMeta.from_json(meta)
        remove_classes = []
        if self.classes_whitelist:
            for obj_class in meta.obj_classes:
                if obj_class.name not in self.classes_whitelist:
                    remove_classes.append(obj_class.name)
            if remove_classes:
                meta = meta.delete_obj_classes(remove_classes)
        return meta
