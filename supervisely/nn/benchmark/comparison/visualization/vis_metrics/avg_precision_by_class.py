from supervisely.nn.benchmark.comparison.evaluation_result import EvalResult
from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
)


class AveragePrecisionByClass(BaseVisMetric):
    MARKDOWN_CLASS_AP = "markdown_class_ap"

    def get_figure(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        labels = dict(r="Average Precision", theta="Class")
        for eval_result in self.eval_results:
            # AP per-class
            ap_per_class = eval_result.mp.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
            ap_per_class[ap_per_class == -1] = 0  # -1 is a placeholder for no GT

            trace_name = EvalResult.name
            fig.add_trace(
                go.Scatterpolar(
                    r=ap_per_class,
                    theta=eval_result.mp.cat_names,
                    name=trace_name,
                    fill="toself",
                    hovertemplate=trace_name
                    + "<br>"
                    + labels["theta"]
                    + ": %{theta}<br>"
                    + labels["r"]
                    + ": %{r:.2f}<br>"
                    + "<extra></extra>",
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], title=labels["r"]),
                angularaxis=dict(title=labels["theta"]),
            ),
            width=800,
            height=800,
            margin=dict(l=80, r=80, t=0, b=0),
            modebar_add=["resetScale"],
            showlegend=True,
        )

        return fig

    @property
    def markdown_widget(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_CLASS_AP).format(
            self.vis_texts.definitions.average_precision
        )
        MarkdownWidget(name=self.MARKDOWN_CLASS_AP, title="Overview", text=text)

    @property
    def chart_widget(self) -> ChartWidget:
        return ChartWidget(name="chart", figure=self.get_figure(), click_data=None)