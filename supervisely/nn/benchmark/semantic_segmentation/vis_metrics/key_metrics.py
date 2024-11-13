from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class KeyMetrics(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "markdown_header",
            "Key Metrics",
            text=self.vis_texts.markdown_key_metrics,
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("base_metrics_chart", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        metrics = self.eval_result.mp.key_metrics()
        fig.add_trace(
            go.Scatterpolar(
                r=list(metrics.values()) + [list(metrics.values())[0]],
                theta=list(metrics.keys()) + [list(metrics.keys())[0]],
                line_color="blue",
                fill="toself",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            showlegend=False,
            polar=dict(
                radialaxis=dict(
                    range=[0, 100],
                    ticks="outside",
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            dragmode=False,
            margin=dict(l=25, r=25, t=25, b=25),
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
            )
        )
        return fig
