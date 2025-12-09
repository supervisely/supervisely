import random
from collections import defaultdict
from pathlib import Path

import supervisely.nn.benchmark.semantic_segmentation.text_templates as vis_texts
from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.acknowledgement import (
    Acknowledgement,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.classwise_error_analysis import (
    ClasswiseErrorAnalysis,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.confusion_matrix import (
    ConfusionMatrix,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.explore_predictions import (
    ExplorePredictions,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.frequently_confused import (
    FrequentlyConfused,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.iou_eou import (
    IntersectionErrorOverUnion,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.key_metrics import (
    KeyMetrics,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.model_predictions import (
    ModelPredictions,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.overview import Overview
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.renormalized_error_ou import (
    RenormalizedErrorOverUnion,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.speedtest import (
    Speedtest,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    MarkdownWidget,
    SidebarWidget,
)
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta


class SemanticSegmentationVisualizer(BaseVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vis_texts = vis_texts
        self._widgets_created = False
        self.ann_opacity = 0.7

        diff_project_info, diff_dataset_infos, existed = self._get_or_create_diff_project()
        self.eval_result.diff_project_info = diff_project_info
        self.eval_result.diff_dataset_infos = diff_dataset_infos
        self.eval_result.matched_pair_data = {}

        self.gt_project_path = str(Path(self.workdir).parent / "gt_project")
        self.pred_project_path = str(Path(self.workdir).parent / "pred_project")

        self.eval_result.images_map = {}
        self.eval_result.images_by_class = defaultdict(set)
        if not existed:
            self._init_match_data()

        # set filtered project meta
        self.eval_result.filtered_project_meta = self._get_filtered_project_meta(self.eval_result)

        self._get_sample_data_for_gallery()

    @property
    def cv_task(self):
        return CVTask.SEMANTIC_SEGMENTATION

    def _create_widgets(self):
        # Modal Gellery
        self.diff_modal = self._create_diff_modal_table()
        self.explore_modal = self._create_explore_modal_table(
            click_gallery_id=self.diff_modal.id, hover_text="Compare with GT"
        )

        # Notifcation
        self.clickable_label = self._create_clickable_label()

        # overview
        overview = Overview(self.vis_texts, self.eval_result)
        self.header = overview.get_header(self.api.user.get_my_info().login)
        self.overview_md = overview.overview_md

        # key metrics
        key_metrics = KeyMetrics(self.vis_texts, self.eval_result)
        self.key_metrics_md = key_metrics.md
        self.key_metrics_table = key_metrics.table
        self.key_metrics_chart = key_metrics.chart

        # explore predictions
        explore_predictions = ExplorePredictions(
            self.vis_texts, self.eval_result, self.explore_modal, self.diff_modal
        )
        self.explore_predictions_md = explore_predictions.md
        self.explore_predictions_gallery = explore_predictions.gallery(self.ann_opacity)

        # model predictions
        model_predictions = ModelPredictions(self.vis_texts, self.eval_result, self.diff_modal)
        self.model_predictions_md = model_predictions.md
        self.model_predictions_table = model_predictions.table

        # intersection over union
        iou_eou = IntersectionErrorOverUnion(self.vis_texts, self.eval_result)
        self.iou_eou_md = iou_eou.md
        self.iou_eou_chart = iou_eou.chart

        # renormalized error over union
        renorm_eou = RenormalizedErrorOverUnion(self.vis_texts, self.eval_result)
        self.renorm_eou_md = renorm_eou.md
        self.renorm_eou_chart = renorm_eou.chart

        # classwise error analysis
        classwise_error_analysis = ClasswiseErrorAnalysis(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.classwise_error_analysis_md = classwise_error_analysis.md
        self.classwise_error_analysis_chart = classwise_error_analysis.chart

        # confusion matrix
        confusion_matrix = ConfusionMatrix(self.vis_texts, self.eval_result, self.explore_modal)
        self.confusion_matrix_md = confusion_matrix.md
        self.confusion_matrix_chart = confusion_matrix.chart

        # frequently confused
        frequently_confused = FrequentlyConfused(
            self.vis_texts, self.eval_result, self.explore_modal
        )
        self.frequently_confused_md = frequently_confused.md
        self.frequently_confused_chart = None
        if not frequently_confused.is_empty:
            self.frequently_confused_chart = frequently_confused.chart

        # Acknowledgement
        acknowledgement = Acknowledgement(self.vis_texts, self.eval_result)
        self.acknowledgement_md = acknowledgement.md

        # SpeedTest
        speedtest = Speedtest(self.vis_texts, self.eval_result)
        self.speedtest_present = not speedtest.is_empty()
        self.speedtest_multiple_batch_sizes = False
        if self.speedtest_present:
            self.speedtest_md_intro = speedtest.intro_md
            self.speedtest_intro_table = speedtest.intro_table
            self.speedtest_multiple_batch_sizes = speedtest.multiple_batche_sizes()
            if self.speedtest_multiple_batch_sizes:
                self.speedtest_batch_inference_md = speedtest.batch_size_md
                self.speedtest_chart = speedtest.chart

        self._widgets_created = True

    def _create_layout(self):
        if not self._widgets_created:
            self._create_widgets()

        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overview_md),
            (1, self.key_metrics_md),
            (0, self.key_metrics_table),
            (0, self.key_metrics_chart),
            (1, self.explore_predictions_md),
            (0, self.explore_predictions_gallery),
            (1, self.model_predictions_md),
            (0, self.model_predictions_table),
            (1, self.iou_eou_md),
            (0, self.iou_eou_chart),
            (1, self.renorm_eou_md),
            (0, self.renorm_eou_chart),
            (1, self.classwise_error_analysis_md),
            (0, self.clickable_label),
            (0, self.classwise_error_analysis_chart),
            (1, self.confusion_matrix_md),
            (0, self.clickable_label),
            (0, self.confusion_matrix_chart),
            (1, self.frequently_confused_md),
        ]
        if self.frequently_confused_chart is not None:
            is_anchors_widgets.append((0, self.clickable_label))
            is_anchors_widgets.append((0, self.frequently_confused_chart))
        if self.speedtest_present:
            is_anchors_widgets.append((1, self.speedtest_md_intro))
            is_anchors_widgets.append((0, self.speedtest_intro_table))
            if self.speedtest_multiple_batch_sizes:
                is_anchors_widgets.append((0, self.speedtest_batch_inference_md))
                is_anchors_widgets.append((0, self.speedtest_chart))

        is_anchors_widgets.append((0, self.acknowledgement_md))
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

    def _init_match_data(self):
        gt_project = Project(self.gt_project_path, OpenMode.READ)
        pred_project = Project(self.pred_project_path, OpenMode.READ)
        diff_map = {ds.id: ds for ds in self.eval_result.diff_dataset_infos}
        pred_map = {ds.id: ds for ds in self.eval_result.pred_dataset_infos}

        def _get_full_name(ds_id: int, ds_id_map: dict):
            ds_info = ds_id_map[ds_id]
            if ds_info.parent_id is None:
                return ds_info.name
            return f"{_get_full_name(ds_info.parent_id, ds_id_map)}/{ds_info.name}"

        diff_dataset_name_map = {_get_full_name(i, diff_map): ds for i, ds in diff_map.items()}
        pred_dataset_name_map = {_get_full_name(i, pred_map): ds for i, ds in pred_map.items()}

        with self.pbar(
            message="Visualizations: Initializing match data", total=pred_project.total_items
        ) as p:
            for pred_dataset in pred_project.datasets:
                pred_dataset: Dataset
                gt_dataset: Dataset = gt_project.datasets.get(pred_dataset.name)
                try:
                    diff_dataset_info = diff_dataset_name_map[pred_dataset.name]
                    pred_dataset_info = pred_dataset_name_map[pred_dataset.name]
                except KeyError:
                    raise RuntimeError(
                        f"Difference project was not created properly. Dataset {pred_dataset.name} is missing"
                    )

                try:
                    for src_images in self.api.image.get_list_generator(
                        pred_dataset_info.id, force_metadata_for_links=False, batch_size=100
                    ):
                        if len(src_images) == 0:
                            continue
                        dst_images = self.api.image.copy_batch_optimized(
                            pred_dataset_info.id,
                            src_images,
                            diff_dataset_info.id,
                            with_annotations=False,
                            skip_validation=True,
                        )
                        for diff_image_info in dst_images:
                            item_name = diff_image_info.name

                            gt_image_info = gt_dataset.get_image_info(item_name)
                            pred_image_info = pred_dataset.get_image_info(item_name)
                            gt_ann = gt_dataset.get_ann(item_name, gt_project.meta)
                            pred_ann = pred_dataset.get_ann(item_name, pred_project.meta)

                            self._update_match_data(
                                gt_image_info.id,
                                gt_image_info=gt_image_info,
                                pred_image_info=pred_image_info,
                                diff_image_info=diff_image_info,
                                gt_annotation=gt_ann,
                                pred_annotation=pred_ann,
                            )

                            assert item_name not in self.eval_result.images_map

                            self.eval_result.images_map[item_name] = gt_image_info.id

                            for label in pred_ann.labels:
                                self.eval_result.images_by_class[label.obj_class.name].add(
                                    gt_image_info.id
                                )

                        p.update(len(src_images))
                except Exception as e:
                    raise RuntimeError(f"Match data was not created properly. {e}")

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

    def _create_clickable_label(self):
        return MarkdownWidget(name="clickable_label", title="", text=self.vis_texts.clickable_label)

    def _get_filtered_project_meta(self, eval_result) -> ProjectMeta:
        remove_classes = []
        meta = eval_result.pred_project_meta.clone()
        meta = meta.merge(eval_result.gt_project_meta)
        if eval_result.classes_whitelist:
            for obj_class in meta.obj_classes:
                if obj_class.name not in eval_result.classes_whitelist:
                    if obj_class.name not in [eval_result.mp.bg_cls_name, "__bg__"]:
                        remove_classes.append(obj_class.name)
            if remove_classes:
                meta = meta.delete_obj_classes(remove_classes)
        return meta
