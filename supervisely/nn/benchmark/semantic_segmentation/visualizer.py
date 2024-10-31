import supervisely.nn.benchmark.semantic_segmentation.vis_metrics.vis_texts as vis_texts
from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.base_metrics import (
    BaseMetrics,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.overview import Overview
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    GalleryWidget,
    MarkdownWidget,
    SidebarWidget,
)


class SemanticSegmentationVisualizer(BaseVisualizer):
    def __init__(self, api, eval_results, workdir="./visualizations"):
        super().__init__(api, eval_results, workdir)

        self.vis_texts = vis_texts
        self._speedtest_present = self.eval_result.speedtest_info is not None
        self._widgets_created = False

    def _create_widgets(self):
        # Modal Gellery
        self.diff_modal_table = self._create_diff_modal_table()
        self.explore_modal_table = self._create_explore_modal_table(self.diff_modal_table.id)

        overview = Overview(self.vis_texts, self.eval_result)
        self.header = overview.get_header(self.api.user.get_my_info().login)
        self.overview_md = overview.overview_md

        base_metrics = BaseMetrics(self.vis_texts, self.eval_result)
        self.base_metrics_chart = base_metrics.chart

        self._widgets_created = True

    def _create_layout(self):
        if not self._widgets_created:
            self._create_widgets()

        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overview_md),
            (0, self.base_metrics_chart),
        ]
        if self._speedtest_present:
            is_anchors_widgets.extend(
                [
                    # SpeedTest
                    # (1, self.speedtest_md_intro),
                    # (0, self.speedtest_intro_table),
                    # (0, self.speed_inference_time_md),
                    # (0, self.speed_inference_time_table),
                    # (0, self.speed_fps_md),
                    # (0, self.speed_fps_table),
                    # (0, self.speed_batch_inference_md),
                    # (0, self.speed_chart),
                ]
            )
        anchors = []
        for is_anchor, widget in is_anchors_widgets:
            if is_anchor:
                anchors.append(widget.id)

        sidebar = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
        layout = ContainerWidget(
            widgets=[sidebar],
            name="main_container",
        )
        return layout

    def _create_explore_modal_table(self, columns_number=3):
        # TODO: table for each evaluation?
        all_predictions_modal_gallery = GalleryWidget(
            "all_predictions_modal_gallery", is_modal=True, columns_number=columns_number
        )
        all_predictions_modal_gallery.set_project_meta(self.eval_result.dt_project_meta)
        return all_predictions_modal_gallery

    def _create_diff_modal_table(self, columns_number=3) -> GalleryWidget:
        diff_modal_gallery = GalleryWidget(
            "diff_predictions_modal_gallery", is_modal=True, columns_number=columns_number
        )
        return diff_modal_gallery
