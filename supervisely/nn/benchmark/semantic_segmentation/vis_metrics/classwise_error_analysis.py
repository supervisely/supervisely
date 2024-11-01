from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
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
            text=self.vis_texts.markdown_eou_per_class,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("classwise_error_analysis", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # # Classwise Segmentation Error Analysis figure
        bar_data, labels = self.eval_result.mp.classwise_segm_error_data
        color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

        for i, column in enumerate(bar_data.columns):
            fig.add_trace(
                go.Bar(
                    name=column,
                    y=bar_data[column],
                    x=labels,
                    marker_color=color_palette[i],
                )
            )

        fig.update_layout(barmode="stack", xaxis_title="Class")
        fig.update_yaxes(range=[0, 1])

        return fig
