from __future__ import annotations

from typing import TYPE_CHECKING

from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class IOUDistribution(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        title = "Localization Accuracy (IoU)"
        if self._loader.cv_task in [CVTask.INSTANCE_SEGMENTATION, CVTask.SEMANTIC_SEGMENTATION]:
            title = "Mask Accuracy (IoU)"
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_localization_accuracy=Widget.Markdown(
                title=title,
                is_header=True,
                formats=[self._loader.vis_texts.definitions.iou_score],
            ),
            markdown_iou_distribution=Widget.Markdown(
                title="IoU Distribution",
                is_header=True,
                formats=[self._loader.vis_texts.definitions.iou_score],
            ),
            notification_avg_iou=Widget.Notification(
                formats_title=[self._loader.base_metrics()["iou"].round(2)]
            ),
            chart=Widget.Chart(),
            collapse_iou=Widget.Collapse(
                Schema(
                    self._loader.vis_texts,
                    markdown_iou_calculation=Widget.Markdown(title="How IoU is calculated?"),
                )
            ),
        )

    def get_figure(self, widget: Widget):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        nbins = 40
        fig.add_trace(go.Histogram(x=self._loader.mp.ious, nbinsx=nbins))
        fig.update_layout(
            # title="IoU Distribution",
            xaxis_title="IoU",
            yaxis_title="Count",
            width=600,
            height=500,
        )

        # Add annotation for mean IoU as vertical line
        mean_iou = self._loader.mp.ious.mean()
        y1 = len(self._loader.mp.ious) // nbins
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
