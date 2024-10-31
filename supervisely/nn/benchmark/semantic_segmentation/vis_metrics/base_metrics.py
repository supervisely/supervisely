from supervisely.nn.benchmark.comparison.detection_visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class BaseMetrics(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        header = MarkdownWidget("markdown_header", "Header", text="## Key metrics")
        return header

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("base_metrics_chart", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        # from plotly.subplots import make_subplots

        fig = go.Figure()

        # Basic segmentation metrics figure
        metrics = {
            "mPixel accuracy": self.eval_result.mp.pixel_accuracy,
            "mPrecision": self.eval_result.mp.precision,
            "mRecall": self.eval_result.mp.recall,
            "mF1-score": self.eval_result.mp.f1_score,
            "mIoU": self.eval_result.mp.iou,
            "mBoundaryIoU": self.eval_result.mp.boundary_iou,
            "mPixel accuracy": self.eval_result.mp.pixel_accuracy,
        }
        fig.add_trace(
            go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                # fill="toself",
                # fillcolor="cornflowerblue",
                line_color="blue",
                # opacity=0.1,
                text=list(metrics.values()),
                textposition=[
                    "bottom right",
                    "top center",
                    "top center",
                    "middle left",
                    "bottom center",
                    "bottom right",
                    "bottom right",
                ],
                textfont=dict(color="blue"),
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showline=False, showticklabels=False, range=[0, 100])
            ),
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            yaxis=dict(showticklabels=False),
        )
        return fig
