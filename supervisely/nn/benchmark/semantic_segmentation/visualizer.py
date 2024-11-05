import supervisely.nn.benchmark.semantic_segmentation.text_templates as vis_texts
from supervisely.nn.benchmark.base_visualizer import BaseVisualizer
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.acknowledgement import (
    Acknowledgement,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.classwise_error_analysis import (
    ClasswiseErrorAnalysis,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.confusion_matrix import (
    ConfusionMatrix,
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
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.overview import Overview
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.renormalized_error_ou import (
    RenormalizedErrorOverUnion,
)
from supervisely.nn.benchmark.semantic_segmentation.vis_metrics.speedtest import (
    Speedtest,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ContainerWidget,
    GalleryWidget,
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
        # self.diff_modal_table = self._create_diff_modal_table()
        # self.explore_modal_table = self._create_explore_modal_table(self.diff_modal_table.id)

        # overview
        overview = Overview(self.vis_texts, self.eval_result)
        self.header = overview.get_header(self.api.user.get_my_info().login)
        self.overview_md = overview.overview_md

        # key metrics
        key_metrics = KeyMetrics(self.vis_texts, self.eval_result)
        self.key_metrics_md = key_metrics.md
        self.key_metrics_chart = key_metrics.chart

        # TODO: Explore predictions

        # intersection over union
        iou_eou = IntersectionErrorOverUnion(self.vis_texts, self.eval_result)
        self.iou_eou_md = iou_eou.md
        self.iou_eou_chart = iou_eou.chart

        # renormalized error over union
        renorm_eou = RenormalizedErrorOverUnion(self.vis_texts, self.eval_result)
        self.renorm_eou_md = renorm_eou.md
        self.renorm_eou_chart = renorm_eou.chart

        # classwise error analysis
        classwise_error_analysis = ClasswiseErrorAnalysis(self.vis_texts, self.eval_result)
        self.classwise_error_analysis_md = classwise_error_analysis.md
        self.classwise_error_analysis_chart = classwise_error_analysis.chart

        # confusion matrix
        confusion_matrix = ConfusionMatrix(self.vis_texts, self.eval_result)
        self.confusion_matrix_md = confusion_matrix.md
        self.confusion_matrix_chart = confusion_matrix.chart

        # frequently confused
        frequently_confused = FrequentlyConfused(self.vis_texts, self.eval_result)
        self.frequently_confused_md = frequently_confused.md
        self.frequently_confused_chart = frequently_confused.chart

        # Acknowledgement
        acknowledgement = Acknowledgement(self.vis_texts, self.eval_result)
        self.acknowledgement_md = acknowledgement.md

        # SpeedTest
        self.speedtest_present = False
        self.speedtest_multiple_batch_sizes = False
        speedtest = Speedtest(self.vis_texts, self.eval_result)
        if not speedtest.is_empty():
            self.speedtest_present = True
            self.speedtest_md_intro = speedtest.intro_md
            self.speedtest_intro_table = speedtest.intro_table
            if speedtest.multiple_batche_sizes():
                self.speedtest_multiple_batch_sizes = True
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
            (0, self.key_metrics_chart),
            # TODO: Explore predictions
            (1, self.iou_eou_md),
            (0, self.iou_eou_chart),
            (1, self.renorm_eou_md),
            (0, self.renorm_eou_chart),
            (1, self.classwise_error_analysis_md),
            (0, self.classwise_error_analysis_chart),
            (1, self.confusion_matrix_md),
            (0, self.confusion_matrix_chart),
            (1, self.frequently_confused_md),
            (0, self.frequently_confused_chart),
        ]
        if self._speedtest_present:
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
