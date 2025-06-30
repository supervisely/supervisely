import random
from pathlib import Path
from typing import List

import supervisely.nn.benchmark.object_detection.text_templates as vis_texts
from supervisely._utils import batched
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagApplicableTo, TagMeta, TagValueType
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import ApiField
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.object_detection.vis_metrics import (
    ConfidenceDistribution,
    ConfidenceScore,
    ConfusionMatrix,
    ExplorePredictions,
    F1ScoreAtDifferentIOU,
    FrequentlyConfused,
    IOUDistribution,
    KeyMetrics,
    ModelPredictions,
    OutcomeCounts,
    Overview,
    PerClassAvgPrecision,
    PerClassOutcomeCounts,
    PRCurve,
    PRCurveByClass,
    Precision,
    Recall,
    RecallVsPrecision,
    ReliabilityDiagram,
    Speedtest,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    MarkdownWidget,
    SidebarWidget,
)
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class ObjectDetectionVisualizer(BaseVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vis_texts = vis_texts
        self._widgets = False
        self.ann_opacity = 0.4

        diff_project_info, diff_dataset_infos, _ = self._get_or_create_diff_project()
        self.eval_result.diff_project_info = diff_project_info
        self.eval_result.diff_dataset_infos = diff_dataset_infos
        self.eval_result.matched_pair_data = {}

        self.gt_project_path = str(Path(self.workdir).parent / "gt_project")
        self.pred_project_path = str(Path(self.workdir).parent / "pred_project")
        self.update_diff_annotations()

        # set filtered project meta
        self.eval_result.filtered_project_meta = self._get_filtered_project_meta(self.eval_result)
        self._get_sample_data_for_gallery()

    @property
    def cv_task(self):
        return CVTask.OBJECT_DETECTION

    def _create_widgets(self):
        # get cv task
        # Modal Gellery
        self.diff_modal = self._create_diff_modal_table()
        self.explore_modal = self._create_explore_modal_table(
            click_gallery_id=self.diff_modal.id, hover_text="Compare with GT"
        )

        # Notifcation
        self.clickable_label = self._create_clickable_label()

        # Overview
        me = self.api.user.get_my_info()
        overview = Overview(self.vis_texts, self.eval_result)
        self.header = overview.get_header(me.login)
        self.overview_md = overview.md

        # IOU Per Class (optional)
        self.iou_per_class_md = overview.iou_per_class_md
        self.iou_per_class_table = overview.iou_per_class_table

        # Key Metrics
        key_metrics = KeyMetrics(self.vis_texts, self.eval_result)
        self.key_metrics_md = key_metrics.md
        self.key_metrics_table = key_metrics.table
        self.overview_chart = key_metrics.chart
        self.custom_ap_description = key_metrics.custom_ap_description_md

        # Explore Predictions
        explore_predictions = ExplorePredictions(
            self.vis_texts, self.eval_result, self.explore_modal, self.diff_modal
        )
        self.explore_predictions_md = explore_predictions.md
        self.explore_predictions_gallery = explore_predictions.gallery(opacity=self.ann_opacity)

        # Model Predictions
        model_predictions = ModelPredictions(self.vis_texts, self.eval_result, self.diff_modal)
        self.model_predictions_md = model_predictions.md
        self.model_predictions_table = model_predictions.table

        # Outcome Counts
        outcome_counts = OutcomeCounts(self.vis_texts, self.eval_result, self.explore_modal)
        self.outcome_counts_md = outcome_counts.md
        self.outcome_counts_chart = outcome_counts.chart

        # Recall
        recall = Recall(self.vis_texts, self.eval_result, self.explore_modal)
        self.recall_md = recall.md
        self.recall_notificaiton = recall.notification
        self.recall_per_class_md = recall.per_class_md
        self.recall_chart = recall.chart

        # Precision
        precision = Precision(self.vis_texts, self.eval_result, self.explore_modal)
        self.precision_md = precision.md
        self.precision_notification = precision.notification
        self.precision_per_class_md = precision.per_class_md
        self.precision_chart = precision.chart

        # RecallVsPrecision
        recall_vs_precision = RecallVsPrecision(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.recall_vs_precision_md = recall_vs_precision.md
        self.recall_vs_precision_chart = recall_vs_precision.chart

        # PRCurve
        pr_curve = PRCurve(self.vis_texts, self.eval_result)
        self.pr_curve_md = pr_curve.md
        self.pr_curve_notificaiton = pr_curve.notification
        self.pr_curve_chart = pr_curve.chart
        self.pr_curve_collapse = pr_curve.collapse

        # PRCurveByClass
        pr_curve_by_class = PRCurveByClass(self.vis_texts, self.eval_result, self.explore_modal)
        self.pr_curve_by_class_md = pr_curve_by_class.md
        self.pr_curve_by_class_chart = pr_curve_by_class.chart

        # ConfusionMatrix
        confusion_matrix = ConfusionMatrix(self.vis_texts, self.eval_result, self.explore_modal)
        self.confusion_matrix_md = confusion_matrix.md
        self.confusion_matrix_chart = confusion_matrix.chart

        # FrequentlyConfused
        frequently_confused = FrequentlyConfused(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.frequently_confused_present = frequently_confused.is_empty is False
        if self.frequently_confused_present:
            self.frequently_confused_md = frequently_confused.md
            self.frequently_confused_chart = frequently_confused.chart
        else:
            self.frequently_confused_md = frequently_confused.empty_md

        # IOUDistribution
        iou_distribution = IOUDistribution(self.vis_texts, self.eval_result)
        if self.cv_task in [CVTask.INSTANCE_SEGMENTATION, CVTask.SEMANTIC_SEGMENTATION]:
            iou_distribution.md_title = "Mask Accuracy (IoU)"
        self.iou_distribution_md = iou_distribution.md
        self.iou_distribution_md_iou_distribution = iou_distribution.md_iou_distribution
        self.iou_distribution_notification = iou_distribution.notification
        self.iou_distribution_chart = iou_distribution.chart

        # ReliabilityDiagram
        reliability_diagram = ReliabilityDiagram(self.vis_texts, self.eval_result)
        self.reliability_diagram_md_calibration_score = reliability_diagram.md_calibration_score
        self.reliability_diagram_collapse_1 = reliability_diagram.collapse_tip
        self.reliability_diagram_md_calibration_score_2 = reliability_diagram.md_calibration_score_2
        self.reliability_diagram_md_reliability_diagram = reliability_diagram.md_reliability_diagram
        self.reliability_diagram_notification = reliability_diagram.notification
        self.reliability_diagram_chart = reliability_diagram.chart
        self.reliability_diagram_collapse_2 = reliability_diagram.collapse

        # ConfidenceScore
        confidence_score = ConfidenceScore(self.vis_texts, self.eval_result)
        self.confidence_score_md_confidence_score = confidence_score.md_confidence_score
        self.confidence_score_notification = confidence_score.notification
        self.confidence_score_chart = confidence_score.chart
        self.confidence_score_md_confidence_score_2 = confidence_score.md_confidence_score_2
        self.confidence_score_collapse_conf_score = confidence_score.collapse_conf_score
        self.confidence_score_md_confidence_score_3 = confidence_score.md_confidence_score_3

        # F1ScoreAtDifferentIOU
        f1_score_at_different_iou = F1ScoreAtDifferentIOU(self.vis_texts, self.eval_result)
        self.f1_score_at_different_iou_md = f1_score_at_different_iou.md
        self.f1_score_at_different_iou_chart = f1_score_at_different_iou.chart

        # ConfidenceDistribution
        confidence_distribution = ConfidenceDistribution(self.vis_texts, self.eval_result)
        self.confidence_distribution_md = confidence_distribution.md
        self.confidence_distribution_chart = confidence_distribution.chart

        # PerClassAvgPrecision
        per_class_avg_precision = PerClassAvgPrecision(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.per_class_avg_precision_md = per_class_avg_precision.md
        self.per_class_avg_precision_chart = per_class_avg_precision.chart

        # PerClassOutcomeCounts
        per_class_outcome_counts = PerClassOutcomeCounts(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.per_class_outcome_counts_md = per_class_outcome_counts.md
        self.per_class_outcome_counts_md_2 = per_class_outcome_counts.md_2
        self.per_class_outcome_counts_collapse = per_class_outcome_counts.collapse
        self.per_class_outcome_counts_chart = per_class_outcome_counts.chart

        # Speedtest init here for overview
        speedtest = Speedtest(self.vis_texts, self.eval_result)
        self.speedtest_present = not speedtest.is_empty()
        self.speedtest_batch_sizes_cnt = speedtest.num_batche_sizes
        if self.speedtest_present:
            self.speedtest_md_intro = speedtest.intro_md
            self.speedtest_table_md = speedtest.table_md
            self.speedtest_table = speedtest.table
            if self.speedtest_batch_sizes_cnt > 1:
                self.speedtest_chart_md = speedtest.chart_md
                self.speedtest_chart = speedtest.chart

        self._widgets = True

    def _create_layout(self):
        if not self._widgets:
            self._create_widgets()

        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overview_md),
        ]

        if self.iou_per_class_table is not None:
            is_anchors_widgets += [
                (0, self.iou_per_class_md),
                (0, self.iou_per_class_table),
            ]

        is_anchors_widgets += [
            # KeyMetrics
            (1, self.key_metrics_md),
            (0, self.key_metrics_table),
        ]

        if self.custom_ap_description is not None:
            is_anchors_widgets.append((0, self.custom_ap_description))

        is_anchors_widgets += [
            (0, self.overview_chart),
            # ExplorePredictions
            (1, self.explore_predictions_md),
            (0, self.explore_predictions_gallery),
            # ModelPredictions
            (1, self.model_predictions_md),
            (0, self.model_predictions_table),
            # OutcomeCounts
            (1, self.outcome_counts_md),
            (0, self.clickable_label),
            (0, self.outcome_counts_chart),
            # Recall
            (1, self.recall_md),
            (0, self.recall_notificaiton),
            (0, self.recall_per_class_md),
            (0, self.clickable_label),
            (0, self.recall_chart),
            # Precision
            (1, self.precision_md),
            (0, self.precision_notification),
            (0, self.precision_per_class_md),
            (0, self.clickable_label),
            (0, self.precision_chart),
            # RecallVsPrecision
            (1, self.recall_vs_precision_md),
            (0, self.clickable_label),
            (0, self.recall_vs_precision_chart),
            # PRCurve
            (1, self.pr_curve_md),
            (0, self.pr_curve_notificaiton),
            (0, self.pr_curve_chart),
            (0, self.pr_curve_collapse),
            # PRCurveByClass
            (0, self.pr_curve_by_class_md),
            (0, self.clickable_label),
            (0, self.pr_curve_by_class_chart),
            # ConfusionMatrix
            (1, self.confusion_matrix_md),
            (0, self.clickable_label),
            (0, self.confusion_matrix_chart),
            # FrequentlyConfused
            (1, self.frequently_confused_md),
        ]
        if self.frequently_confused_present:
            is_anchors_widgets.append((0, self.clickable_label))
            is_anchors_widgets.append((0, self.frequently_confused_chart))

        is_anchors_widgets.extend(
            [
                # IOUDistribution
                (1, self.iou_distribution_md),
                (0, self.iou_distribution_md_iou_distribution),
                (0, self.iou_distribution_notification),
                (0, self.iou_distribution_chart),
                # ReliabilityDiagram
                (1, self.reliability_diagram_md_calibration_score),
                (0, self.reliability_diagram_collapse_1),
                (0, self.reliability_diagram_md_calibration_score_2),
                (1, self.reliability_diagram_md_reliability_diagram),
                (0, self.reliability_diagram_notification),
                (0, self.reliability_diagram_chart),
                (0, self.reliability_diagram_collapse_2),
                # ConfidenceScore
                (1, self.confidence_score_md_confidence_score),
                (0, self.confidence_score_notification),
                (0, self.confidence_score_chart),
                (0, self.confidence_score_md_confidence_score_2),
                (0, self.confidence_score_collapse_conf_score),
                (0, self.confidence_score_md_confidence_score_3),
                # F1ScoreAtDifferentIOU
                (1, self.f1_score_at_different_iou_md),
                (0, self.f1_score_at_different_iou_chart),
                # ConfidenceDistribution
                (1, self.confidence_distribution_md),
                (0, self.confidence_distribution_chart),
                # PerClassAvgPrecision
                (1, self.per_class_avg_precision_md),
                (0, self.clickable_label),
                (0, self.per_class_avg_precision_chart),
                # PerClassOutcomeCounts
                (1, self.per_class_outcome_counts_md),
                (0, self.per_class_outcome_counts_md_2),
                (0, self.per_class_outcome_counts_collapse),
                (0, self.clickable_label),
                (0, self.per_class_outcome_counts_chart),
            ]
        )

        if self.speedtest_present:
            # SpeedTest
            is_anchors_widgets.append((1, self.speedtest_md_intro))
            is_anchors_widgets.append((0, self.speedtest_table_md))
            is_anchors_widgets.append((0, self.speedtest_table))
            if self.speedtest_batch_sizes_cnt > 1:
                is_anchors_widgets.append((0, self.speedtest_chart_md))
                is_anchors_widgets.append((0, self.speedtest_chart))
        anchors = []
        for is_anchor, widget in is_anchors_widgets:
            if is_anchor:
                anchors.append(widget.id)

        sidebar = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
        layout = ContainerWidget(
            widgets=[sidebar, self.explore_modal, self.diff_modal],
            name="main_container",
        )
        return layout

    def _create_clickable_label(self):
        return MarkdownWidget(name="clickable_label", title="", text=self.vis_texts.clickable_label)

    def update_diff_annotations(self):
        pred_project_id = self.eval_result.pred_project_id
        pred_project_meta = self.eval_result.pred_project_meta
        meta = self._update_pred_meta_with_tags(pred_project_id, pred_project_meta)
        self.eval_result.pred_project_meta = meta

        self._update_diff_meta(meta)

        self._add_tags_to_pred_project(
            self.eval_result.mp.matches, self.eval_result.pred_project_id
        )

        gt_project = Project(self.gt_project_path, OpenMode.READ)
        pred_project = Project(self.pred_project_path, OpenMode.READ)
        diff_dataset_id_map = {ds.id: ds for ds in self.eval_result.diff_dataset_infos}
        logger.info(f"Diff datasets names: {[ds.name for ds in diff_dataset_id_map.values()]}")

        def _get_full_name(ds_id: int):
            ds_info = diff_dataset_id_map[ds_id]
            if ds_info.parent_id is None:
                return ds_info.name
            return f"{_get_full_name(ds_info.parent_id)}/{ds_info.name}"

        diff_dataset_name_map = {_get_full_name(i): ds for i, ds in diff_dataset_id_map.items()}

        matched_id_map = self._get_matched_id_map()  # dt_id -> gt_id
        matched_gt_ids = set(matched_id_map.values())

        outcome_tag = meta.get_tag_meta("outcome")
        conf_meta = meta.get_tag_meta("confidence")
        if conf_meta is None:
            conf_meta = meta.get_tag_meta("conf")
        match_tag = meta.get_tag_meta("matched_gt_id")

        pred_tag_list = []
        with self.pbar(
            message="Visualizations: Creating difference project", total=pred_project.total_items
        ) as progress:
            logger.debug(
                "Creating diff project data",
                extra={
                    "pred_project": [ds.name for ds in pred_project.datasets],
                    "gt_project": [ds.name for ds in gt_project.datasets],
                },
            )
            for pred_dataset in pred_project.datasets:
                pred_dataset: Dataset
                gt_dataset: Dataset = gt_project.datasets.get(pred_dataset.name)
                diff_dataset_info = diff_dataset_name_map[pred_dataset.name]
                for batch_names in batched(pred_dataset.get_items_names(), 100):
                    diff_anns = []
                    gt_image_ids = []
                    pred_img_ids = []
                    for item_name in batch_names:
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

                            if conf < self.eval_result.mp.conf_threshold:
                                continue  # do not add labels with low confidence to diff project
                            if match_tag_id:
                                continue  # do not add TP labels to diff project
                            label = label.add_tag(Tag(outcome_tag, value))
                            label = label.add_tag(Tag(match_tag, int(label.geometry.sly_id)))
                            labels.append(label)

                        # FN
                        for label in gt_ann.labels:
                            if self.eval_result.classes_whitelist:
                                if label.obj_class.name not in self.eval_result.classes_whitelist:
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
                        self._update_match_data(
                            gt_image_info.id,
                            gt_image_info=gt_image_info,
                            pred_image_info=pred_image_info,
                            gt_annotation=gt_ann,
                            pred_annotation=pred_ann,
                            diff_annotation=diff_ann,
                        )

                    diff_img_infos = self.api.image.copy_batch(diff_dataset_info.id, pred_img_ids)
                    ids = [img.id for img in diff_img_infos]
                    self.api.annotation.upload_anns(ids, diff_anns, progress_cb=progress.update)
                    for gt_img_id, diff_img_info in zip(gt_image_ids, diff_img_infos):
                        self._update_match_data(gt_img_id, diff_image_info=diff_img_info)

        with self.pbar(
            message="Visualizations: Append tags to predictions", total=len(pred_tag_list)
        ) as p:
            self.api.image.tag.add_to_objects(
                self.eval_result.pred_project_id, pred_tag_list, progress=p
            )

    def _init_match_data(self):
        gt_project = Project(self.gt_project_path, OpenMode.READ)
        pred_project = Project(self.pred_project_path, OpenMode.READ)
        diff_dataset_id_map = {ds.id: ds for ds in self.eval_result.diff_dataset_infos}
        logger.info(f"Diff datasets names: {[ds.name for ds in diff_dataset_id_map.values()]}")

        def _get_full_name(ds_id: int):
            ds_info = diff_dataset_id_map[ds_id]
            if ds_info.parent_id is None:
                return ds_info.name
            return f"{_get_full_name(ds_info.parent_id)}/{ds_info.name}"

        diff_dataset_name_map = {_get_full_name(i): ds for i, ds in diff_dataset_id_map.items()}

        meta_json = self.api.project.get_meta(self.eval_result.diff_project_info.id)
        self.eval_result.diff_project_meta = ProjectMeta.from_json(meta_json)

        with self.pbar(
            message="Visualizations: Initializing match data", total=pred_project.total_items
        ) as p:
            for pred_dataset in pred_project.datasets:
                pred_dataset: Dataset
                gt_dataset: Dataset = gt_project.datasets.get(pred_dataset.name)
                try:
                    diff_dataset_info = diff_dataset_name_map[pred_dataset.name]
                except KeyError:
                    raise RuntimeError(
                        f"Difference project was not created properly. Dataset {pred_dataset.name} is missing"
                    )

                for item_names_batch in batched(pred_dataset.get_items_names(), 50):
                    # diff project may be not created yet
                    item_names_batch.sort()
                    try:
                        diff_img_infos_batch: List[ImageInfo] = sorted(
                            self.api.image.get_list(
                                diff_dataset_info.id,
                                filters=[
                                    {
                                        ApiField.FIELD: ApiField.NAME,
                                        ApiField.OPERATOR: "in",
                                        ApiField.VALUE: item_names_batch,
                                    }
                                ],
                                force_metadata_for_links=False,
                            ),
                            key=lambda x: x.name,
                        )
                        diff_anns_batch_dict = {
                            ann_info.image_id: Annotation.from_json(
                                ann_info.annotation, self.eval_result.diff_project_meta
                            )
                            for ann_info in self.api.annotation.download_batch(
                                diff_dataset_info.id,
                                [img_info.id for img_info in diff_img_infos_batch],
                                force_metadata_for_links=False,
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

                            self._update_match_data(
                                gt_image_info.id,
                                gt_image_info=gt_image_info,
                                pred_image_info=pred_image_info,
                                diff_image_info=diff_img_info,
                                gt_annotation=gt_ann,
                                pred_annotation=pred_ann,
                                diff_annotation=diff_ann,
                            )

                        p.update(len(item_names_batch))
                    except Exception:
                        raise RuntimeError("Difference project was not created properly")

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

        meta = self.api.project.update_meta(project_id, meta)
        return meta

    def _update_diff_meta(self, meta: ProjectMeta):
        new_obj_classes = []
        for obj_class in meta.obj_classes:
            new_obj_classes.append(obj_class.clone(geometry_type=AnyGeometry))
        meta = meta.clone(obj_classes=new_obj_classes)
        self.eval_result.diff_project_meta = self.api.project.update_meta(
            self.eval_result.diff_project_info.id, meta
        )

    def _add_tags_to_pred_project(self, matches: list, pred_project_id: int):

        # get tag metas
        # outcome_tag_meta = self.dt_project_meta.get_tag_meta("outcome")
        match_tag_meta = self.eval_result.pred_project_meta.get_tag_meta("matched_gt_id")
        iou_tag_meta = self.eval_result.pred_project_meta.get_tag_meta("iou")

        # mappings
        gt_ann_mapping = self.eval_result.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.eval_result.click_data.dt_id_mapper.map_obj

        # add tags to objects
        logger.info("Adding tags to DT project")

        with self.pbar(
            message="Visualizations: Adding tags to predictions", total=len(matches)
        ) as p:
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

                self.api.image.tag.add_to_objects(pred_project_id, pred_tag_list)
                p.update(len(batch))

    def _get_matched_id_map(self):
        gt_ann_mapping = self.eval_result.click_data.gt_id_mapper.map_obj
        dt_ann_mapping = self.eval_result.click_data.dt_id_mapper.map_obj
        dtId2matched_gt_id = {}
        for match in self.eval_result.mp.matches_filtered:
            if match["type"] == "TP":
                dtId2matched_gt_id[dt_ann_mapping[match["dt_id"]]] = gt_ann_mapping[match["gt_id"]]
        return dtId2matched_gt_id

    def _is_label_compatible_to_cv_task(self, label: Label):
        if self.cv_task == CVTask.OBJECT_DETECTION:
            return isinstance(label.geometry, Rectangle)
        elif self.cv_task == CVTask.INSTANCE_SEGMENTATION:
            return isinstance(label.geometry, (Bitmap, Polygon))
        elif self.cv_task == CVTask.SEMANTIC_SEGMENTATION:
            return isinstance(label.geometry, Bitmap)
        return False

    def _get_sample_data_for_gallery(self):
        """Get sample images with annotations for visualization (preview gallery)"""
        sample_images = []
        limit = 9
        for ds_info in self.eval_result.pred_dataset_infos:
            images = self.api.image.get_list(
                ds_info.id, limit=limit, force_metadata_for_links=False
            )
            sample_images.extend(images)
        if len(sample_images) > limit:
            sample_images = random.sample(sample_images, limit)
        self.eval_result.sample_images = sample_images
        ids = [img.id for img in sample_images]
        self.eval_result.sample_anns = self.api.annotation.download_batch(ds_info.id, ids)
