import datetime
from collections import defaultdict
from pathlib import Path

import supervisely.nn.benchmark.comparison.semantic_segmentation.text_templates as vis_texts
from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
from supervisely.nn.benchmark.comparison.semantic_segmentation.vis_metrics import (  # AveragePrecisionByClass,; CalibrationScore,; ExplorePredictions,; LocalizationAccuracyIoU,; OutcomeCounts,; PrCurve,; PrecisionRecallF1,; Speedtest,
    Overview,
)
from supervisely.nn.benchmark.visualization.renderer import Renderer
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    GalleryWidget,
    MarkdownWidget,
    SidebarWidget,
)
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta


class SemanticSegmentationComparisonVisualizer:
    def __init__(self, comparison):
        self.comparison = comparison
        self.api = comparison.api
        self.vis_texts = vis_texts
        self.ann_opacity = 0.7
        self.eval_results = comparison.eval_results
        self.gt_project_info = None
        self.gt_project_meta = None
        # self._widgets_created = False

        for eval_result in self.eval_results:
            eval_result.api = self.api  # add api to eval_result for overview widget
            self._get_eval_project_infos(eval_result)

        self._create_widgets()
        layout = self._create_layout()

        self.renderer = Renderer(layout, str(Path(self.comparison.workdir, "visualizations")))

    def visualize(self):
        return self.renderer.visualize()

    def upload_results(self, team_id: int, remote_dir: str, progress=None):
        return self.renderer.upload_results(self.api, team_id, remote_dir, progress)

    def _create_widgets(self):
        # Modal Gellery
        self.diff_modal = self._create_diff_modal_table()
        self.explore_modal = self._create_explore_modal_table(
            click_gallery_id=self.diff_modal.id, hover_text="Compare with GT"
        )

        # Notifcation
        self.clickable_label = self._create_clickable_label()

        # Speedtest init here for overview
        # speedtest = Speedtest(self.vis_texts, self.comparison.eval_results)

        # Overview
        overview = Overview(self.vis_texts, self.comparison.eval_results)
        overview.team_id = self.comparison.team_id
        self.header = self._create_header()
        self.overviews = self._create_overviews(overview)
        self.overview_md = overview.overview_md
        self.key_metrics_md = self._create_key_metrics()
        # self.key_metrics_table = overview.get_table_widget(
        #     latency=speedtest.latency, fps=speedtest.fps
        # )
        self.overview_chart = overview.chart_widget

        # # SpeedTest
        # self.speedtest_present = False
        # if not speedtest.is_empty():
        #     self.speedtest_present = True
        #     self.speedtest_md_intro = speedtest.md_intro
        #     self.speedtest_intro_table = speedtest.intro_table
        #     self.speed_inference_time_md = speedtest.inference_time_md
        #     self.speed_inference_time_table = speedtest.inference_time_table
        #     self.speed_fps_md = speedtest.fps_md
        #     self.speed_fps_table = speedtest.fps_table
        #     self.speed_batch_inference_md = speedtest.batch_inference_md
        #     self.speed_chart = speedtest.chart

    def _create_layout(self):
        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overview_md),
            (0, self.overviews),
            (1, self.key_metrics_md),
            # (0, self.key_metrics_table),
            (0, self.overview_chart),
            # Explore Predictions
            # (1, self.explore_predictions_md),
            # (0, self.explore_predictions_gallery),
        ]
        # if self.speedtest_present:
        #     is_anchors_widgets.extend(
        #         [
        #             # SpeedTest
        #             (1, self.speedtest_md_intro),
        #             (0, self.speedtest_intro_table),
        #             (0, self.speed_inference_time_md),
        #             (0, self.speed_inference_time_table),
        #             (0, self.speed_fps_md),
        #             (0, self.speed_fps_table),
        #             (0, self.speed_batch_inference_md),
        #             (0, self.speed_chart),
        #         ]
        #     )
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

    def _create_header(self) -> MarkdownWidget:
        me = self.api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_main_text = " ∣ ".join(  #  vs. or | or ∣
            eval_res.name for eval_res in self.comparison.eval_results
        )
        header_text = self.vis_texts.markdown_header.format(header_main_text, me, current_date)
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    def _create_overviews(self, vm: Overview) -> ContainerWidget:
        grid_cols = 2
        overview_widgets = vm.overview_widgets
        if len(overview_widgets) > 2:
            grid_cols = 3
        if len(overview_widgets) % 4 == 0:
            grid_cols = 4
        return ContainerWidget(
            overview_widgets,
            name="overview_container",
            title="Overview",
            grid=True,
            grid_cols=grid_cols,
        )

    def _create_key_metrics(self) -> MarkdownWidget:
        key_metrics_text = self.vis_texts.markdown_key_metrics.format(
            self.vis_texts.definitions.average_precision,
            self.vis_texts.definitions.confidence_threshold,
            self.vis_texts.definitions.confidence_score,
        )
        return MarkdownWidget("markdown_key_metrics", "Key Metrics", text=key_metrics_text)

    def _create_outcome_counts_md(self) -> MarkdownWidget:
        outcome_counts_text = self.vis_texts.markdown_outcome_counts.format(
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
            self.vis_texts.definitions.false_negatives,
        )
        return MarkdownWidget("markdown_outcome_counts", "Outcome Counts", text=outcome_counts_text)

    def _create_outcome_counts_diff_md(self) -> MarkdownWidget:
        outcome_counts_text = self.vis_texts.markdown_outcome_counts_diff
        return MarkdownWidget(
            "markdown_outcome_counts_diff", "Outcome Counts Differences", text=outcome_counts_text
        )

    def _create_explore_modal_table(
        self, columns_number=3, click_gallery_id=None, hover_text=None
    ) -> GalleryWidget:
        gallery = GalleryWidget(
            "all_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            click_gallery_id=click_gallery_id,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].pred_project_meta)
        if hover_text:
            gallery.add_image_left_header(hover_text)
        return gallery

    def _create_diff_modal_table(self, columns_number=3) -> GalleryWidget:
        gallery = GalleryWidget(
            "diff_predictions_modal_gallery",
            is_modal=True,
            columns_number=columns_number,
            opacity=self.ann_opacity,
        )
        gallery.set_project_meta(self.eval_results[0].pred_project_meta)
        return gallery

    def _create_clickable_label(self):
        return MarkdownWidget("clickable_label", "", text=self.vis_texts.clickable_label)

    def _get_eval_project_infos(self, eval_result):
        # get project infos
        if self.gt_project_info is None:
            self.gt_project_info = self.api.project.get_info_by_id(eval_result.gt_project_id)
        eval_result.gt_project_info = self.gt_project_info
        eval_result.pred_project_info = self.api.project.get_info_by_id(eval_result.pred_project_id)

        # get project metas
        if self.gt_project_meta is None:
            self.gt_project_meta = ProjectMeta.from_json(
                self.api.project.get_meta(eval_result.gt_project_id)
            )
        eval_result.gt_project_meta = self.gt_project_meta
        eval_result.pred_project_meta = ProjectMeta.from_json(
            self.api.project.get_meta(eval_result.pred_project_id)
        )

        # eval_result.pred_dataset_infos = self.api.dataset.get_list(
        #     eval_result.pred_project_id, recursive=True
        # )
