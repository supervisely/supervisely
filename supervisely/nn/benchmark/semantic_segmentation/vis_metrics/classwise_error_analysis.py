from supervisely.nn.benchmark.comparison.detection_visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ClasswiseErrorAnalysis(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "classwise_error_analysis",
            "Classwise Segmentation Error Analysis",
            text="## Classwise Segmentation Error Analysis",
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("intersection_error_over_union", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # # Classwise Segmentation Error Analysis figure
        bar_data, labels = self.eval_result.mp.classwise_segm_error_data
        color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

        for data, label, color in zip(bar_data, labels, color_palette):
            fig.add_trace(
                go.Bar(
                    x=bar_data.columns,
                    y=data,
                    name=label,
                    marker=dict(color=color),
                )
            )

        fig.update_layout(xaxis_title="Class", yaxis_title="Error")

        return fig
