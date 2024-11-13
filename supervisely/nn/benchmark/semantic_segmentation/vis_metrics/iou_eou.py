from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class IntersectionErrorOverUnion(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

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
                marker=dict(colors=["cornflowerblue", "moccasin", "lightgreen", "orangered"]),
            )
        )

        return fig