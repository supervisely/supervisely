import numpy as np

from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    TableWidget,
)


class LocalizationAccuracyIoU(BaseVisMetrics):
    @property
    def header_md(self) -> MarkdownWidget:
        title = "Localization Accuracy (IoU)"
        if self.eval_results[0].cv_task in [
            CVTask.INSTANCE_SEGMENTATION,
            CVTask.SEMANTIC_SEGMENTATION,
        ]:
            title = "Mask Accuracy (IoU)"
        text_template = self.vis_texts.markdown_localization_accuracy
        text = text_template.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(
            name="markdown_localization_accuracy",
            title=title,
            text=text,
        )

    @property
    def iou_distribution_md(self) -> MarkdownWidget:
        text_template = self.vis_texts.markdown_iou_distribution
        text = text_template.format(self.vis_texts.definitions.iou_score)
        return MarkdownWidget(
            name="markdown_iou_distribution",
            title="IoU Distribution",
            text=text,
        )

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "Avg. IoU"]
        res["content"] = []
        for i, eval_result in enumerate(self.eval_results, 1):
            value = round(eval_result.mp.base_metrics()["iou"], 2)
            model_name = f"[{i}] {eval_result.name}"
            row = [model_name, value]
            dct = {
                "row": row,
                "id": model_name,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [{"disableSort": True}, {"disableSort": True}]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(
            name="localization_accuracy_table", data=res, show_header_controls=False, fix_columns=1
        )

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(name="chart_iou_distribution", figure=self.get_figure())

    @property
    def collapse_tip(self) -> CollapseWidget:
        inner_md = MarkdownWidget(
            name="markdown_iou_calculation",
            title="How IoU is calculated?",
            text=self.vis_texts.markdown_iou_calculation,
        )
        return CollapseWidget(widgets=[inner_md])

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error
        from scipy.stats import gaussian_kde  # pylint: disable=import-error

        fig = go.Figure()
        nbins = 40
        # min_value = min([r.mp.ious[0] for r in self.eval_results])
        x_range = np.linspace(0.5, 1, 500)
        hist_data = [np.histogram(r.mp.ious, bins=nbins) for r in self.eval_results]
        bin_width = min([bin_edges[1] - bin_edges[0] for _, bin_edges in hist_data])

        for i, (eval_result, (hist, bin_edges)) in enumerate(zip(self.eval_results, hist_data)):
            name = f"[{i+1}] {eval_result.name}"
            kde = gaussian_kde(eval_result.mp.ious)
            density = kde(x_range)

            scaling_factor = len(eval_result.mp.ious) * bin_width
            scaled_density = density * scaling_factor

            fig.add_trace(
                go.Bar(
                    x=bin_edges[:-1],
                    y=hist,
                    width=bin_width,
                    name=f"{name} (Bars)",
                    offset=0,
                    opacity=0.2,
                    hovertemplate=name + "<br>IoU: %{x:.2f}<br>Count: %{y}<extra></extra>",
                    marker=dict(color=eval_result.color, line=dict(width=0)),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=scaled_density,
                    name=f"{name} (KDE)",
                    line=dict(color=eval_result.color, width=2),
                    hovertemplate=name + "<br>IoU: %{x:.2f}<br>Count: %{y:.1f}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="IoU",
            yaxis_title="Count",
        )

        # Add annotation for mean IoU as vertical line
        for i, eval_result in enumerate(self.eval_results):
            mean_iou = eval_result.mp.ious.mean()
            y1 = len(eval_result.mp.ious) // nbins
            fig.add_shape(
                type="line",
                x0=mean_iou,
                x1=mean_iou,
                y0=0,
                y1=y1,
                line=dict(color="orange", width=2, dash="dash"),
            )

        fig.update_layout(
            barmode="overlay",
            bargap=0,
            bargroupgap=0,
            dragmode=False,
            yaxis=dict(rangemode="tozero"),
            xaxis=dict(range=[0.5, 1]),
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
            ),
        )
        return fig
