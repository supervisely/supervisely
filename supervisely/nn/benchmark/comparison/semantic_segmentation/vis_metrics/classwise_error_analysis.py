from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ClasswiseErrorAnalysis(BaseVisMetrics):
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
        return ChartWidget("classwise_error_analysis", self.get_figure())

    def get_figure(self):
        import numpy as np
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        # Цветовая палитра для метрик
        color_palette = ["#8ACAA1", "#FFE4B5", "#F7ADAA", "#dd3f3f"]
        metrics = ["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]

        group_width = 0.7

        for model_idx, eval_result in enumerate(self.eval_results):
            bar_data, labels = eval_result.mp.classwise_segm_error_data
            model_name = eval_result.name

            for metric_idx, metric_name in enumerate(metrics):
                # hover_customdata = [f"metric: {metric_name} for class '{l}' ({model_name})" for l in labels]
                hover_customdata = [
                    f"class: {l}<br>model: {model_name}<br>{metric_name}" for l in labels
                ]
                fig.add_trace(
                    go.Bar(
                        name=metric_name,
                        x=np.arange(len(labels)) + model_idx * group_width * 0.3,
                        y=bar_data[metric_name],
                        customdata=hover_customdata,
                        hovertemplate="%{customdata}: %{y:.2f}<extra></extra>",
                        marker=dict(color=color_palette[metric_idx]),
                        width=group_width / len(metrics),
                        offsetgroup=model_idx,
                        base=bar_data[metrics[:metric_idx]].sum(axis=1) if metric_idx > 0 else None,
                    )
                )

        fig.update_layout(
            showlegend=False,
            barmode="stack",
            xaxis=dict(
                title="Classes",
                tickvals=np.arange(len(labels)) + (len(self.eval_results) - 1 - group_width) / 4,
                ticktext=labels,
            ),
            width=800 if len(labels) < 10 else 1000,
        )

        return fig
