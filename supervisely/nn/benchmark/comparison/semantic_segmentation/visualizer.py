import supervisely.nn.benchmark.comparison.semantic_segmentation.text_templates as texts
from supervisely.nn.benchmark.comparison.base_visualizer import BaseComparisonVisualizer
from supervisely.nn.benchmark.comparison.semantic_segmentation.vis_metrics import (
    Overview,
    Speedtest
)
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    MarkdownWidget,
    SidebarWidget,
)


class SemanticSegmentationComparisonVisualizer(BaseComparisonVisualizer):
    vis_texts = texts
    ann_opacity = 0.7

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
        self.overviews = self._create_overviews(overview)
        self.overview_md = overview.overview_md
        self.key_metrics_md = self._create_key_metrics()
        self.key_metrics_table = overview.get_table_widget(
            latency=speedtest.latency, fps=speedtest.fps
        )
        self.overview_chart = overview.chart_widget

        # # SpeedTest
        self.speedtest_present = not speedtest.is_empty()
        if self.speedtest_present:
            self.speedtest_md_intro = speedtest.md_intro
            self.speedtest_intro_table = speedtest.intro_table
            self.speed_inference_time_md = speedtest.inference_time_md
            self.speed_inference_time_table = speedtest.inference_time_table
            self.speed_fps_md = speedtest.fps_md
            self.speed_fps_table = speedtest.fps_table
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
            # (1, self.explore_predictions_md),
            # (0, self.explore_predictions_gallery),
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
                    (0, self.speed_batch_inference_md),
                    (0, self.speed_chart),
                ]
            )
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

    def _create_key_metrics(self) -> MarkdownWidget:
        key_metrics_text = self.vis_texts.markdown_key_metrics.format(
            self.vis_texts.definitions.average_precision,
            self.vis_texts.definitions.confidence_threshold,
            self.vis_texts.definitions.confidence_score,
        )
        return MarkdownWidget("markdown_key_metrics", "Key Metrics", text=key_metrics_text)
