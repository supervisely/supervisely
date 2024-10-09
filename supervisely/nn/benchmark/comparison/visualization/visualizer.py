import datetime
import importlib
from pathlib import Path

from jinja2 import Template

from supervisely.api.api import Api
from supervisely.nn.benchmark.comparison.visualization.vis_metrics import (
    AveragePrecisionByClass,
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
        return api.file.upload_directory(
            team_id, self.output_dir, remote_dir, change_name_if_conflict=False
        )


class ComparisonVisualizer:
    def __init__(self, comparison):
        self.comparison = comparison
        self.api = comparison.api
        self.vis_texts = None
        if self.comparison.task_type == "object detection":
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
        self.header = self._create_header()
        self.overviews = self._create_overviews()
        self.key_metrics = self._create_key_metrics()
        self.overview_chart = self._create_overview_chart()

        # TODO: Explore Predictions

        # Outcome Counts
        self.outcome_counts_md = self._create_outcome_counts_md()
        self.outcome_counts_main = self._create_outcome_counts_main()
        self.outcome_counts_comparison = self._create_outcome_counts_comparison()

        # Precision-Recall Curve
        self.pr_curve_md = self._create_pr_curve_md()
        self.pr_curve_collapsed_widgets = self._create_pr_curve_collapsed_widgets()
        self.pr_curve_notification = self._create_pr_curve_notification()
        self.pr_curve_chart = self._create_pr_curve_chart()

        # Average Precision by Class
        # TODO: Niko
        self.avg_prec_by_class_md = self._create_avg_precision_by_class_md()
        self.avg_prec_by_class_chart = self._create_avg_precision_by_class_chart()

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
            # TODO: Almaz
            (1, self.pr_curve_md),
            (0, self.pr_curve_collapsed_widgets),
            (0, self.pr_curve_notification),
            (0, self.pr_curve_chart),
            # Average Precision by Class
            (1, self.avg_prec_by_class_md),
            (0, self.avg_prec_by_class_chart),
            #
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

    def _create_overviews(self) -> ContainerWidget:
        return ContainerWidget(
            Overview(self.vis_texts, self.comparison.evaluation_results).overview_widgets
        )

    def _create_key_metrics(self) -> MarkdownWidget:
        key_metrics_text = self.vis_texts.markdown_key_metrics.format(
            self.vis_texts.definitions.average_precision,
            self.vis_texts.definitions.confidence_threshold,
            self.vis_texts.definitions.confidence_score,
        )
        return MarkdownWidget("markdown_key_metrics", "Key Metrics", text=key_metrics_text)

    def _create_overview_chart(self) -> ChartWidget:
        chart = Overview(self.vis_texts, self.comparison.evaluation_results).chart_widget
        chart.save_data(
            self.comparison.output_dir
        )  # TODO: maybe save_data should be called once in the end and save all data recursively
        return chart

    def _create_outcome_counts_md(self) -> MarkdownWidget:
        outcome_counts_text = self.vis_texts.markdown_outcome_counts.format(
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
            self.vis_texts.definitions.false_negatives,
        )
        return MarkdownWidget("markdown_outcome_counts", "Outcome Counts", text=outcome_counts_text)

    def _create_outcome_counts_main(self) -> ChartWidget:
        chart = OutcomeCounts(self.vis_texts, self.comparison.evaluation_results).chart_widget_main
        chart.save_data(self.comparison.output_dir)  # TODO: the same as in _create_overview_chart
        return chart

    def _create_outcome_counts_comparison(self) -> ChartWidget:
        chart = OutcomeCounts(
            self.vis_texts, self.comparison.evaluation_results
        ).chart_widget_comparison
        chart.save_data(self.comparison.output_dir)  # TODO: the same as in _create_overview_chart
        return chart

    def _create_avg_precision_by_class_md(self):
        return AveragePrecisionByClass(
            self.vis_texts, self.comparison.evaluation_results
        ).markdown_widget

    def _create_avg_precision_by_class_chart(self):
        return AveragePrecisionByClass(
            self.vis_texts, self.comparison.evaluation_results
        ).chart_widget

    def _create_modal_tables(self):
        # TODO: table for each evaluation?
        all_predictions_modal_gallery = GalleryWidget(
            "all_predictions_modal_gallery", is_modal=True
        )

    def _create_pr_curve_md(self):
        return PrCurve(self.vis_texts, self.comparison.evaluation_results).markdown_widget

    def _create_pr_curve_collapsed_widgets(self):
        return PrCurve(self.vis_texts, self.comparison.evaluation_results).collapsed_widget

    def _create_pr_curve_notification(self):
        return PrCurve(self.vis_texts, self.comparison.evaluation_results).notification_widget

    def _create_pr_curve_chart(self):
        return PrCurve(self.vis_texts, self.comparison.evaluation_results).chart_widget
