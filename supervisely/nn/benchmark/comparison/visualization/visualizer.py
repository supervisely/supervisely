import datetime
import importlib
import json
from pathlib import Path
from typing import Optional

from jinja2 import Template

from supervisely.api.api import Api
from supervisely.io.fs import dir_empty, get_directory_size
from supervisely.nn.benchmark.comparison.visualization.vis_metrics import (
    AveragePrecisionByClass,
    CalibrationScore,
    ExplorePredictions,
    LocalizationAccuracyIoU,
    OutcomeCounts,
    Overview,
    PrCurve,
    PrecisionRecallF1,
    Speedtest,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    BaseWidget,
    ContainerWidget,
    GalleryWidget,
    MarkdownWidget,
    SidebarWidget,
)
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.task.progress import tqdm_sly


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
        state = self.layout.get_state()
        with open(Path(self.output_dir).joinpath("state.json"), "w") as f:
            json.dump(state, f)
        template = self.render()
        with open(Path(self.output_dir).joinpath("template.vue"), "w") as f:
            f.write(template)
        return template

    def visualize(self):
        return self.save()

    def upload_results(
        self, api: Api, team_id: int, remote_dir: str, progress: Optional[tqdm_sly] = None
    ) -> str:
        if dir_empty(self.output_dir):
            raise RuntimeError(
                "No visualizations to upload. You should call visualize method first."
            )
        if progress is None:
            progress = tqdm_sly
        dir_total = get_directory_size(self.output_dir)
        dir_name = Path(remote_dir).name
        with progress(
            message=f"Uploading visualizations to {dir_name}",
            total=dir_total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            remote_dir = api.file.upload_directory(
                team_id,
                self.output_dir,
                remote_dir,
                change_name_if_conflict=True,
                progress_size_cb=pbar.update,
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

    def upload_results(self, team_id: int, remote_dir: str, progress=None):
        return self.viz.upload_results(self.api, team_id, remote_dir, progress)

    def _create_widgets(self):
        # Modal Gellery
        self.diff_modal_table = self._create_diff_modal_table()
        self.explore_modal_table = self._create_explore_modal_table(self.diff_modal_table.id)

        # Notifcation
        self.clickable_label = self._create_clickable_label()

        # Speedtest init here for overview
        speedtest = Speedtest(self.vis_texts, self.comparison.evaluation_results)

        # Overview
        overview = Overview(self.vis_texts, self.comparison.evaluation_results)
        self.header = self._create_header()
        self.overviews = self._create_overviews(overview)
        self.overview_md = overview.overview_md
        self.key_metrics_md = self._create_key_metrics()
        self.key_metrics_table = overview.get_table_widget(
            latency=speedtest.latency, fps=speedtest.fps
        )
        self.overview_chart = overview.chart_widget

        # TODO: Explore Predictions
        columns_number = len(self.comparison.evaluation_results) + 1
        self.explore_predictions_modal_gallery = self._create_explore_modal_table(columns_number)
        explore_predictions = ExplorePredictions(
            self.vis_texts,
            self.comparison.evaluation_results,
            explore_modal_table=self.explore_predictions_modal_gallery,
        )
        self.explore_predictions_md = explore_predictions.difference_predictions_md
        self.explore_predictions_gallery = explore_predictions.explore_gallery
        # self.explore_same_errors_md = explore_predictions.same_errors_md
        # self.explore_same_errors_gallery = explore_predictions.same_errors_gallery

        # Outcome Counts
        outcome_counts = OutcomeCounts(
            self.vis_texts,
            self.comparison.evaluation_results,
            explore_modal_table=self.explore_modal_table,
        )
        self.outcome_counts_md = self._create_outcome_counts_md()
        self.outcome_counts_diff_md = self._create_outcome_counts_diff_md()
        self.outcome_counts_main = outcome_counts.chart_widget_main
        self.outcome_counts_comparison = outcome_counts.chart_widget_comparison

        # Precision-Recall Curve
        pr_curve = PrCurve(self.vis_texts, self.comparison.evaluation_results)
        self.pr_curve_md = pr_curve.markdown_widget
        self.pr_curve_collapsed_widgets = pr_curve.collapsed_widget
        self.pr_curve_table = pr_curve.table_widget
        self.pr_curve_chart = pr_curve.chart_widget

        # Average Precision by Class
        avg_prec_by_class = AveragePrecisionByClass(
            self.vis_texts,
            self.comparison.evaluation_results,
            explore_modal_table=self.explore_modal_table,
        )
        self.avg_prec_by_class_md = avg_prec_by_class.markdown_widget
        self.avg_prec_by_class_chart = avg_prec_by_class.chart_widget

        # Precision, Recall, F1
        precision_recall_f1 = PrecisionRecallF1(
            self.vis_texts,
            self.comparison.evaluation_results,
            explore_modal_table=self.explore_modal_table,
        )
        self.precision_recall_f1_md = precision_recall_f1.markdown_widget
        self.precision_recall_f1_table = precision_recall_f1.table_widget
        self.precision_recall_f1_chart = precision_recall_f1.chart_main_widget
        self.precision_per_class_title_md = precision_recall_f1.precision_per_class_title_md
        self.precision_per_class_chart = precision_recall_f1.chart_precision_per_class_widget
        self.recall_per_class_title_md = precision_recall_f1.recall_per_class_title_md
        self.recall_per_class_chart = precision_recall_f1.chart_recall_per_class_widget
        self.f1_per_class_chart = precision_recall_f1.chart_f1_per_class_widget
        self.f1_per_class_title_md = precision_recall_f1.f1_per_class_title_md

        # Classification Accuracy
        # TODO: ???

        # Localization Accuracy (IoU)
        loc_acc = LocalizationAccuracyIoU(self.vis_texts, self.comparison.evaluation_results)
        self.loc_acc_header_md = loc_acc.header_md
        self.loc_acc_iou_distribution_md = loc_acc.iou_distribution_md
        self.loc_acc_chart = loc_acc.chart
        self.loc_acc_table = loc_acc.table_widget

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

        # SpeedTest
        self.speedtest_present = False
        if not speedtest.is_empty():
            self.speedtest_present = True
            self.speedtest_md_intro = speedtest.md_intro
            self.speedtest_intro_table = speedtest.intro_table
            self.speed_overview_md = speedtest.md_speed_overview
            self.speed_md_overview_table = speedtest.md_speed_overview_table
            self.speed_overview_table = speedtest.speed_overview_table
            self.speed_md_fps_table = speedtest.md_fps_table
            self.speed_fps_table = speedtest.fps_table
            self.speed_batch_inference = speedtest.md_batch_inference
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
            # Explore Predictions # TODO
            (1, self.explore_predictions_md),
            (0, self.explore_predictions_gallery),
            # Outcome Counts
            (1, self.outcome_counts_md),
            (0, self.outcome_counts_main),
            (0, self.outcome_counts_diff_md),
            (0, self.outcome_counts_comparison),
            # Precision-Recall Curve
            (1, self.pr_curve_md),
            (0, self.pr_curve_collapsed_widgets),
            (0, self.pr_curve_table),
            (0, self.pr_curve_chart),
            # Average Precision by Class
            (1, self.avg_prec_by_class_md),
            (0, self.avg_prec_by_class_chart),
            # Precision, Recall, F1
            (1, self.precision_recall_f1_md),
            (0, self.precision_recall_f1_table),
            (0, self.clickable_label),
            (0, self.precision_recall_f1_chart),
            (0, self.precision_per_class_title_md),
            (0, self.precision_per_class_chart),
            (0, self.recall_per_class_title_md),
            (0, self.recall_per_class_chart),
            (0, self.f1_per_class_title_md),
            (0, self.f1_per_class_chart),
            # Classification Accuracy # TODO
            # Localization Accuracy (IoU)
            (1, self.loc_acc_header_md),
            (0, self.loc_acc_table),
            (0, self.loc_acc_iou_distribution_md),
            (0, self.loc_acc_chart),
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
        if self.speedtest_present:
            is_anchors_widgets.extend(
                [
                    # SpeedTest
                    (1, self.speedtest_md_intro),
                    (0, self.speedtest_intro_table),
                    (0, self.speed_overview_md),
                    (0, self.speed_md_overview_table),
                    (0, self.speed_overview_table),
                    (0, self.speed_md_fps_table),
                    (0, self.speed_fps_table),
                    (0, self.speed_batch_inference),
                    (0, self.speed_chart),
                ]
            )
        anchors = []
        for is_anchor, widget in is_anchors_widgets:
            if is_anchor:
                anchors.append(widget.id)

        sidebar = SidebarWidget(widgets=[i[1] for i in is_anchors_widgets], anchors=anchors)
        layout = ContainerWidget(
            widgets=[sidebar, self.explore_modal_table, self.explore_predictions_modal_gallery],
            name="main_container",
        )
        return layout

    def _create_header(self) -> MarkdownWidget:
        me = self.api.user.get_my_info().login
        current_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        header_main_text = " ∣ ".join(  #  vs. or | or ∣
            eval_res.name for eval_res in self.comparison.evaluation_results
        )
        header_text = self.vis_texts.markdown_header.format(header_main_text, me, current_date)
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    def _create_overviews(self, vm: Overview) -> ContainerWidget:
        grid_cols = 2
        if len(vm.overview_widgets) > 2:
            grid_cols = 3
        if len(vm.overview_widgets) % 4 == 0:
            grid_cols = 4
        return ContainerWidget(
            vm.overview_widgets,
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
        outcome_counts_text = self.vis_texts.markdown_outcome_counts_diff.format(
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
            self.vis_texts.definitions.false_negatives,
        )
        return MarkdownWidget(
            "markdown_outcome_counts_diff", "Outcome Counts Differences", text=outcome_counts_text
        )

    def _create_explore_modal_table(self, columns_number=3):
        # TODO: table for each evaluation?
        all_predictions_modal_gallery = GalleryWidget(
            "all_predictions_modal_gallery", is_modal=True, columns_number=columns_number
        )
        all_predictions_modal_gallery.set_project_meta(
            self.comparison.evaluation_results[0].dt_project_meta
        )
        return all_predictions_modal_gallery

    def _create_diff_modal_table(self, columns_number=3) -> GalleryWidget:
        diff_modal_gallery = GalleryWidget(
            "diff_predictions_modal_gallery", is_modal=True, columns_number=columns_number
        )
        return diff_modal_gallery

    def _create_clickable_label(self):
        return MarkdownWidget("clickable_label", "", text=self.vis_texts.clickable_label)
