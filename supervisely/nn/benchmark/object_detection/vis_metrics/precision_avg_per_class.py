from __future__ import annotations

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class PerClassAvgPrecision(DetectionVisMetric):
    MARKDOWN = "per_class_avg_precision"
    CHART = "per_class_avg_precision"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_class_ap
        text = text.format(self.vis_texts.definitions.average_precision)
        return MarkdownWidget(self.MARKDOWN, "Average Precision by Class", text)

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget(self.CHART, self._get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].theta}`,",
        )
        return chart

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        # AP per-class
        ap_per_class = self.eval_result.mp.AP_per_class()
        labels = dict(r="Average Precision", theta="Class")
        fig = px.scatter_polar(
            r=ap_per_class,
            theta=self.eval_result.mp.cat_names,
            # title="Per-class Average Precision (AP)",
            labels=labels,
            width=800,
            height=800,
            range_r=[0, 1],
        )
        fig.update_traces(fill="toself")
        fig.update_layout(
            modebar_add=["resetScale"],
            margin=dict(l=80, r=80, t=0, b=0),
        )
        fig.update_traces(
            hovertemplate=labels["theta"]
            + ": %{theta}<br>"
            + labels["r"]
            + ": %{r:.2f}<br>"
            + "<extra></extra>"
        )
        return fig
