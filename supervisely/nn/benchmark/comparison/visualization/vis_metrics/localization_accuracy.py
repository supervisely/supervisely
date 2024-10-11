from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
)
from supervisely.nn.benchmark.cv_tasks import CVTask


class LocalizationAccuracyIoU(BaseVisMetric):
    @property
    def header_md(self) -> MarkdownWidget:
        title = "Localization Accuracy (IoU)"
        if self.eval_results[0].cv_task in [
            CVTask.INSTANCE_SEGMENTATION,
            CVTask.SEMANTIC_SEGMENTATION,
        ]:
            title = "Mask Accuracy (IoU)"
        text_template = self.vis_texts.markdown_localization_accuracy
        text = text_template.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(
            name="markdown_localization_accuracy",
            title=title,
            text=text,
        )

    @property
    def iou_distribution_md(self) -> MarkdownWidget:
        text_template = self.vis_texts.markdown_iou_distribution
        text = text_template.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(
            name="markdown_iou_distribution",
            title="IoU Distribution",
            text=text,
        )

    @property
    def notification(self) -> NotificationWidget:
        description = "<br>".join(
            f"{ev.name}: {ev.mp.base_metrics()['iou']:.2f}" for ev in self.eval_results
        )
        return NotificationWidget(name="notification_avg_iou", title="Avg. IoU", desc=description)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name="chart_iou_distribution", figure=self.get_figure(), click_data=None)

    @property
    def collapse_tip(self) -> CollapseWidget:
        inner_md = MarkdownWidget(
            name="markdown_iou_calculation",
            title="How IoU is calculated?",
            text=self.vis_texts.markdown_iou_calculation,
        )
        return CollapseWidget(widgets=[inner_md])

    def get_figure(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        nbins = 40
        for i, eval_result in enumerate(self.eval_results):
            name = f"[{i+1}]{eval_result.name}"
            fig.add_trace(
                go.Histogram(
                    x=eval_result.mp.ious,
                    nbinsx=nbins,
                    name=name,
                    hovertemplate=name + "<br>IoU: %{x:.2f}<br>Count: %{y}<extra></extra>",
                )
            )

        fig.update_layout(
            # title="IoU Distribution",
            xaxis_title="IoU",
            yaxis_title="Count",
            width=600,
            height=500,
        )

        # Add annotation for mean IoU as vertical line
        for i, eval_result in enumerate(self.eval_results):
            mean_iou = eval_result.mp.ious.mean()
            y1 = len(eval_result.mp.ious) // nbins
            fig.add_shape(
                type="line",
                x0=mean_iou,
                x1=mean_iou,
                y0=0,
                y1=y1,
                line=dict(color="orange", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=mean_iou,
                y=y1,
                text=f"[{i+1}] {eval_result.name}<br>Mean IoU: {mean_iou:.2f}",
                showarrow=False,
            )

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
