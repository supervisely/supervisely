from supervisely.nn.benchmark.comparison.detection_visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class FrequentlyConfused(BaseVisMetric):
    def __init__(self, vis_texts, eval_result: SemanticSegmentationEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "frequently_confused",
            "Frequency Confused Classes",
            text="## Frequency Confused Classes",
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("frequently_confused", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # Frequency of Confused Classes figure
        probs, indexes_2d = self.eval_result.mp.frequently_confused
        confused_classes = []
        for idx in indexes_2d:
            gt_idx, pred_idx = idx[0], idx[1]
            gt_class = self.eval_result.mp.eval_data.index[gt_idx]
            pred_class = self.eval_result.mp.eval_data.index[pred_idx]
            confused_classes.append(f"{gt_class}-{pred_class}")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=confused_classes,
                y=probs,
                orientation="v",
                text=probs,
            )
        )
        # fig.update_traces(
        #     textposition="outside",
        #     marker=dict(color=probs, colorscale="orrd"),
        # )
        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            yaxis_range=[0, max(probs) + 0.1],
            yaxis=dict(showticklabels=False),
            font=dict(size=24),
        )

        return fig
