from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class IntersectionErrorOverUnion(SemanticSegmVisMetric):

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "intersection_error_over_union",
            "Intersection & Error Over Union",
            text=self.vis_texts.markdown_iou,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("intersection_error_over_union", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # Intersection & Error Over Union figure
        labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
        values = [
            self.eval_result.mp.iou,
            self.eval_result.mp.boundary_eou,
            self.eval_result.mp.extent_eou,
            self.eval_result.mp.segment_eou,
        ]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textposition="outside",
                textinfo="percent+label",
                marker=dict(colors=["#8ACAA1", "#FFE4B5", "#F7ADAA", "#dd3f3f"]),
            )
        )

        return fig
