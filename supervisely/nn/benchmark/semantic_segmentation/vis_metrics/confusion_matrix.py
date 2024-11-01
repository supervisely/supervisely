from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ConfusionMatrix(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "confusion_matrix",
            "Confusion Matrix",
            text=self.vis_texts.markdown_confusion_matrix,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("confusion_matrix", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # # Confusion Matrix figure
        confusion_matrix, class_names = self.eval_result.mp.confusion_matrix

        x = class_names
        y = x[::-1].copy()
        text_anns = [[str(el) for el in row] for row in confusion_matrix]

        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                x=x,
                y=y,
                colorscale="orrd",
                showscale=False,
                text=text_anns,
                hoverinfo="text",
            )
        )

        fig.update_layout(xaxis_title="Predicted", yaxis_title="True")

        return fig
