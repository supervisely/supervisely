import datetime
import importlib
import os
from pathlib import Path

from jinja2 import Template

from supervisely.api.api import Api
from supervisely.io.fs import dir_empty
from supervisely.nn.benchmark.comparison.visualization.vis_metrics import (
    AveragePrecisionByClass,
    CalibrationScore,
    LocalizationAccuracyIoU,
    OutcomeCounts,
    Overview,
    PrCurve,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    BaseWidget,
    ChartWidget,
    ContainerWidget,
    GalleryWidget,
    MarkdownWidget,
    SidebarWidget,
)
from supervisely.nn.benchmark.cv_tasks import CVTask


class BaseVisualizer:

    def __init__(self, template: str, layout: BaseWidget, output_dir: str) -> None:
        self.main_template = template
        self.layout = layout
        self.output_dir = output_dir

    @property
    def _template_data(self):
        return {"layout": self.layout.to_html()}

    def render(self):
        return Template(self.main_template).render(self._template_data)

    def get_state(self):
        return {}

    def save(self) -> None:
        self.layout.save_data(self.output_dir)
        self.layout.save_state(self.output_dir)
        template = self.render()
        with open(Path(self.output_dir).joinpath("template.vue"), "w") as f:
            f.write(template)
        return template

    def visualize(self):
        return self.save()

    def upload_results(self, api: Api, team_id: int, remote_dir: str) -> str:
        if dir_empty(self.output_dir):
            raise RuntimeError(
                "No visualizations to upload. You should call visualize method first."
            )
        remote_dir = api.file.upload_directory(
            team_id, self.output_dir, remote_dir, change_name_if_conflict=True
        )
        src = self.save_report_link(api, team_id, remote_dir)
        api.file.upload(team_id=team_id, src=src, dst=remote_dir.rstrip("/") + "/open.lnk")
        return remote_dir

    def save_report_link(self, api: Api, team_id: int, remote_dir: str):
        report_link = self.get_report_link(api, team_id, remote_dir)
        pth = Path(self.output_dir).joinpath("open.lnk")
        with open(pth, "w") as f:
            f.write(report_link)
        return str(pth)

    def get_report_link(self, api: Api, team_id: int, remote_dir: str):
        template_path = remote_dir.rstrip("/") + "/" + "template.vue"
        vue_template_info = api.file.get_info_by_path(team_id, template_path)

        report_link = "/model-benchmark?id=" + str(vue_template_info.id)
        return report_link


class ComparisonVisualizer:
    def __init__(self, comparison):
        self.comparison = comparison
        self.api = comparison.api
        self.vis_texts = None
        if self.comparison.task_type == CVTask.OBJECT_DETECTION:
            import supervisely.nn.benchmark.comparison.visualization.vis_metrics.text_templates as vis_texts  # noqa

            self.vis_texts = vis_texts
        else:
            self.vis_texts = importlib.import_module(
                "supervisely.nn.benchmark.comparison.visualization.vis_metrics.text_templates"
            )  # TODO: change for other task types

        template_path = Path(__file__).parent.joinpath("comparison_template.html")
        template = template_path.read_text()

        self._create_widgets()
        layout = self._create_layout()

        self.viz = BaseVisualizer(template, layout, self.comparison.output_dir)

    def visualize(self):
        return self.viz.visualize()

    def upload_results(self, team_id: int, remote_dir: str):
        return self.viz.upload_results(self.api, team_id, remote_dir)

    def _create_widgets(self):
        # TODO: add modal galleries
        # Overview
        overview = Overview(self.vis_texts, self.comparison.evaluation_results)
        self.header = self._create_header()
        self.overviews = self._create_overviews(overview)
        self.key_metrics = self._create_key_metrics()
        self.overview_chart = overview.chart_widget

        # TODO: Explore Predictions

        # Outcome Counts
        outcome_counts = OutcomeCounts(self.vis_texts, self.comparison.evaluation_results)
        self.outcome_counts_md = self._create_outcome_counts_md()
        self.outcome_counts_main = outcome_counts.chart_widget_main
        self.outcome_counts_comparison = outcome_counts.chart_widget_comparison

        # Precision-Recall Curve
        pr_curve = PrCurve(self.vis_texts, self.comparison.evaluation_results)
        self.pr_curve_md = pr_curve.markdown_widget
        self.pr_curve_collapsed_widgets = pr_curve.collapsed_widget
        self.pr_curve_table = pr_curve.table_widget
        self.pr_curve_notification = pr_curve.notification_widget
        self.pr_curve_chart = pr_curve.chart_widget

        # Average Precision by Class
        avg_prec_by_class = AveragePrecisionByClass(
            self.vis_texts, self.comparison.evaluation_results
        )
        self.avg_prec_by_class_md = avg_prec_by_class.markdown_widget
        self.avg_prec_by_class_chart = avg_prec_by_class.chart_widget

        # Precision, Recall, F1
        # TODO: Almaz

        # Classification Accuracy
        # TODO: ???

        # Localization Accuracy (IoU)
        loc_acc = LocalizationAccuracyIoU(self.vis_texts, self.comparison.evaluation_results)
        self.loc_acc_header_md = loc_acc.header_md
        self.loc_acc_iou_distribution_md = loc_acc.iou_distribution_md
        self.loc_acc_notification = loc_acc.notification
        self.loc_acc_chart = loc_acc.chart
        self.loc_acc_collapse_tip = loc_acc.collapse_tip

        # Calibration Score
        cal_score = CalibrationScore(self.vis_texts, self.comparison.evaluation_results)
        self.cal_score_md = cal_score.header_md
        self.cal_score_md_2 = cal_score.header_md_2
        self.cal_score_collapse_tip = cal_score.collapse_tip
        self.cal_score_table = cal_score.table
        self.cal_score_reliability_diagram_md = cal_score.reliability_diagram_md
        self.cal_score_reliability_chart = cal_score.reliability_chart
        self.cal_score_collapse_ece = cal_score.collapse_ece
        self.cal_score_confidence_score_md = cal_score.confidence_score_md
        self.cal_score_confidence_chart = cal_score.confidence_chart
        self.cal_score_confidence_score_md_2 = cal_score.confidence_score_md_2
        self.cal_score_collapse_conf_score = cal_score.collapse_conf_score

    def _create_layout(self):
        is_anchors_widgets = [
            # Overview
            (0, self.header),
            (1, self.overviews),
            (1, self.key_metrics),
            (0, self.overview_chart),
            # Explore Predictions # TODO
            # Outcome Counts
            (1, self.outcome_counts_md),
            (0, self.outcome_counts_main),
            (0, self.outcome_counts_comparison),
            # Precision-Recall Curve
            (1, self.pr_curve_md),
            (0, self.pr_curve_collapsed_widgets),
            (0, self.pr_curve_table),
            (0, self.pr_curve_notification),
            (0, self.pr_curve_chart),
            # Average Precision by Class
            (1, self.avg_prec_by_class_md),
            (0, self.avg_prec_by_class_chart),
            # Precision, Recall, F1 # TODO
            # Classification Accuracy # TODO
            # Localization Accuracy (IoU)
            (1, self.loc_acc_header_md),
            (1, self.loc_acc_iou_distribution_md),
            (0, self.loc_acc_notification),
            (0, self.loc_acc_chart),
            (0, self.loc_acc_collapse_tip),
            # Calibration Score
            (1, self.cal_score_md),
            (0, self.cal_score_md_2),
            (0, self.cal_score_collapse_tip),
            (0, self.cal_score_table),
            (1, self.cal_score_reliability_diagram_md),
            (0, self.cal_score_reliability_chart),
            (0, self.cal_score_collapse_ece),
            (1, self.cal_score_confidence_score_md),
            (0, self.cal_score_confidence_chart),
            (0, self.cal_score_confidence_score_md_2),
            (0, self.cal_score_collapse_conf_score),
        ]
        anchors = []
        for is_anchor, widget in is_anchors_widgets:
            if is_anchor:
                anchors.append(widget.id)

        layout = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
        return layout

    def _create_header(self) -> MarkdownWidget:
        me = self.api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_main_text = " vs. ".join(
            eval_res.name for eval_res in self.comparison.evaluation_results
        )
        header_text = self.vis_texts.markdown_header.format(header_main_text, me, current_date)
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    def _create_overviews(self, vm: Overview) -> ContainerWidget:
        return ContainerWidget(vm.overview_widgets)

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

    def _create_modal_tables(self):
        # TODO: table for each evaluation?
        all_predictions_modal_gallery = GalleryWidget(
            "all_predictions_modal_gallery", is_modal=True
        )
