from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class RenormalizedErrorOverUnion(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "renormalized_error_over_union",
            "Renormalized Error over Union",
            text=self.vis_texts.markdown_renormalized_error_ou,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("intersection_error_over_union", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # Renormalized Error over Union figure
        labels = ["boundary", "extent", "segment"]
        values = [
            self.eval_result.mp.boundary_renormed_eou,
            self.eval_result.mp.extent_renormed_eou,
            self.eval_result.mp.segment_renormed_eou,
        ]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                orientation="v",
                text=values,
                width=[0.5, 0.5, 0.5],
                textposition="outside",
                marker_color=["moccasin", "lightgreen", "orangered"],
            )
        )

        return fig