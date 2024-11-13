from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
)


class IOUDistribution(DetectionVisMetric):
    MARKDOWN_LOCALIZATION_ACCURACY = "localization_accuracy"
    MARKDOWN_IOU_DISTRIBUTION = "iou_distribution"
    NOTIFICATION = "iou_distribution"
    COLLAPSE = "iou_distribution"
    CHART = "iou_distribution"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.md_title = "Localization Accuracy (IoU)"

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_localization_accuracy
        text = text.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(self.MARKDOWN_LOCALIZATION_ACCURACY, self.md_title, text)

    @property
    def md_iou_distribution(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_iou_distribution
        text = text.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(self.MARKDOWN_IOU_DISTRIBUTION, self.md_title, text)

    @property
    def notification(self) -> NotificationWidget:
        title, _ = self.vis_texts.notification_avg_iou.values()
        return NotificationWidget(
            self.NOTIFICATION,
            title.format(self.eval_result.mp.base_metrics()["iou"].round(2)),
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    @property
    def collapse(self) -> CollapseWidget:
        md1 = MarkdownWidget(
            "iou_calculation",
            "How IoU is calculated?",
            self.vis_texts.markdown_iou_calculation,
        )
        md2 = MarkdownWidget(
            "what_is_pr_curve",
            "How the PR curve is built?",
            self.vis_texts.markdown_what_is_pr_curve.format(
                self.vis_texts.definitions.confidence_score,
                self.vis_texts.definitions.true_positives,
                self.vis_texts.definitions.false_positives,
            ),
        )
        return CollapseWidget([md1, md2])

    def _get_figure(self):  # -> go.Figure:
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        nbins = 40
        fig.add_trace(go.Histogram(x=self.eval_result.mp.ious, nbinsx=nbins))
        fig.update_layout(
            xaxis_title="IoU",
            yaxis_title="Count",
            width=600,
            height=500,
        )

        # Add annotation for mean IoU as vertical line
        mean_iou = self.eval_result.mp.ious.mean()
        y1 = len(self.eval_result.mp.ious) // nbins
        fig.add_shape(
            type="line",
            x0=mean_iou,
            x1=mean_iou,
            y0=0,
            y1=y1,
            line=dict(color="orange", width=2, dash="dash"),
        )
        fig.update_traces(hovertemplate="IoU: %{x:.2f}<br>Count: %{y}<extra></extra>")
        fig.add_annotation(x=mean_iou, y=y1, text=f"Mean IoU: {mean_iou:.2f}", showarrow=False)
        fig.update_layout(
            dragmode=False,
            modebar=dict(
                remove=[
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )
        return fig
