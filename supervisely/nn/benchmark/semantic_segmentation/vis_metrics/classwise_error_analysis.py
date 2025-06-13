from typing import Dict

from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ClasswiseErrorAnalysis(SemanticSegmVisMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "classwise_error_analysis",
            "Classwise Segmentation Error Analysis",
            text=self.vis_texts.markdown_eou_per_class,
        )

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget("classwise_error_analysis", self.get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].label}`,",
        )
        return chart

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # # Classwise Segmentation Error Analysis figure
        bar_data, labels = self.eval_result.mp.classwise_segm_error_data
        color_palette = ["#8ACAA1", "#FFE4B5", "#F7ADAA", "#dd3f3f"]

        for i, column in enumerate(bar_data.columns):
            fig.add_trace(
                go.Bar(
                    name=column,
                    y=bar_data[column],
                    x=labels,
                    marker_color=color_palette[i],
                    hovertemplate="Class: %{x}<br> %{name}: %{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(barmode="stack", xaxis_title="Class")
        if len(labels) < 10:
            fig.update_layout(width=800)
        fig.update_yaxes(range=[0, 1])

        return fig
