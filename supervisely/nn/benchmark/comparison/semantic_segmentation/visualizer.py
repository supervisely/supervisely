from typing import List

import supervisely.nn.benchmark.comparison.semantic_segmentation.text_templates as texts
from supervisely.nn.benchmark.comparison.base_visualizer import BaseComparisonVisualizer
from supervisely.nn.benchmark.comparison.semantic_segmentation.vis_metrics import (
    ClasswiseErrorAnalysis,
    ExplorePredictions,
    FrequentlyConfused,
    IntersectionErrorOverUnion,
    Overview,
    RenormalizedErrorOverUnion,
    Speedtest,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    MarkdownWidget,
    SidebarWidget,
)


class SemanticSegmentationComparisonVisualizer(BaseComparisonVisualizer):
    vis_texts = texts
    ann_opacity = 0.7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_results: List[SemanticSegmentationEvalResult]

    def _create_widgets(self):
        # Modal Gellery
        self.diff_modal = self._create_diff_modal_table()
        self.explore_modal = self._create_explore_modal_table(
            click_gallery_id=self.diff_modal.id, hover_text="Compare with GT"
        )

        # Notifcation
        self.clickable_label = self._create_clickable_label()

        # Speedtest init here for overview
        speedtest = Speedtest(self.vis_texts, self.comparison.eval_results)

        # Overview
        overview = Overview(self.vis_texts, self.comparison.eval_results)
        overview.team_id = self.comparison.team_id
        self.header = self._create_header()
        self.overviews = self._create_overviews(overview, grid_cols=2)
        self.overview_md = overview.overview_md
        self.key_metrics_md = self._create_key_metrics()
        self.key_metrics_table = overview.get_table_widget(
            latency=speedtest.latency, fps=speedtest.fps
        )
        self.overview_chart = overview.chart_widget

        # Explore Predictions
        columns_number = len(self.comparison.eval_results) + 1  # +1 for GT
        self.explore_predictions_modal_gallery = self._create_explore_modal_table(columns_number)
        explore_predictions = ExplorePredictions(
            self.vis_texts,
            self.comparison.eval_results,
            explore_modal_table=self.explore_predictions_modal_gallery,
        )
        self.explore_predictions_md = explore_predictions.difference_predictions_md
        self.explore_predictions_gallery = explore_predictions.explore_gallery

        # IntersectionErrorOverUnion
        iou_eou = IntersectionErrorOverUnion(self.vis_texts, self.comparison.eval_results)
        self.iou_eou_md = iou_eou.md
        self.iou_eou_chart = iou_eou.chart

        # RenormalizedErrorOverUnion
        reou = RenormalizedErrorOverUnion(self.vis_texts, self.comparison.eval_results)
        self.reou_md = reou.md
        self.reou_chart = reou.chart

        # ClasswiseErrorAnalysis
        classwise_ea = ClasswiseErrorAnalysis(self.vis_texts, self.comparison.eval_results)
        self.classwise_ea_md = classwise_ea.md
        self.classwise_ea_chart = classwise_ea.chart

        # FrequentlyConfused
        frequently_confused = FrequentlyConfused(self.vis_texts, self.comparison.eval_results)
        self.frequently_confused_md = frequently_confused.md
        self.frequently_confused_chart = frequently_confused.chart

        # # SpeedTest
        self.speedtest_present = not speedtest.is_empty()
        self.speedtest_multiple_batch_sizes = False

        if self.speedtest_present:
            self.speedtest_md_intro = speedtest.md_intro
            self.speedtest_intro_table = speedtest.intro_table
            self.speed_inference_time_md = speedtest.inference_time_md
            self.speed_inference_time_table = speedtest.inference_time_table
            self.speed_fps_md = speedtest.fps_md
            self.speed_fps_table = speedtest.fps_table
            self.speedtest_multiple_batch_sizes = speedtest.multiple_batche_sizes()
            if self.speedtest_multiple_batch_sizes:
                self.speed_batch_inference_md = speedtest.batch_inference_md
                self.speed_chart = speedtest.chart

    def _create_layout(self):
        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overview_md),
            (0, self.overviews),
            (1, self.key_metrics_md),
            (0, self.key_metrics_table),
            (0, self.overview_chart),
            # Explore Predictions
            (1, self.explore_predictions_md),
            (0, self.explore_predictions_gallery),
            # IntersectionErrorOverUnion
            (1, self.iou_eou_md),
            (0, self.iou_eou_chart),
            # RenormalizedErrorOverUnion
            (1, self.reou_md),
            (0, self.reou_chart),
            # ClasswiseErrorAnalysis
            (1, self.classwise_ea_md),
            (0, self.classwise_ea_chart),
            # FrequentlyConfused
            (1, self.frequently_confused_md),
            (0, self.frequently_confused_chart),
        ]
        if self.speedtest_present:
            is_anchors_widgets.extend(
                [
                    # SpeedTest
                    (1, self.speedtest_md_intro),
                    (0, self.speedtest_intro_table),
                    (0, self.speed_inference_time_md),
                    (0, self.speed_inference_time_table),
                    (0, self.speed_fps_md),
                    (0, self.speed_fps_table),
                ]
            )
            if self.speedtest_multiple_batch_sizes:
                is_anchors_widgets.append((0, self.speed_batch_inference_md))
                is_anchors_widgets.append((0, self.speed_chart))
        anchors = []
        for is_anchor, widget in is_anchors_widgets:
            if is_anchor:
                anchors.append(widget.id)

        sidebar = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
        layout = ContainerWidget(
            widgets=[sidebar, self.explore_modal, self.explore_predictions_modal_gallery],
            name="main_container",
        )
        return layout

    def _create_key_metrics(self) -> MarkdownWidget:
        return MarkdownWidget(
            "markdown_key_metrics", "Key Metrics", text=self.vis_texts.markdown_key_metrics
        )
