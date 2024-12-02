from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class RenormalizedErrorOverUnion(SemanticSegmVisMetric):

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
        labels = ["Boundary EoU", "Extent EoU", "Segment EoU"]
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
                marker_color=["#FFE4B5", "#F7ADAA", "#dd3f3f"],
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_traces(hovertemplate="%{x}: %{y:.2f}<extra></extra>")
        fig.update_layout(width=600)

        return fig
