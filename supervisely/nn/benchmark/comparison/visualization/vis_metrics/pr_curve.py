import numpy as np

from supervisely.nn.benchmark.comparison.visualization.vis_metrics.vis_metric import (
    BaseVisMetric,
)
from supervisely.nn.benchmark.comparison.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
    TableWidget,
)


class PrCurve(BaseVisMetric):
    MARKDOWN_PR_CURVE = "markdown_pr_curve"
    MARKDOWN_PR_TRADE_OFFS = "markdown_trade_offs"
    MARKDOWN_WHAT_IS_PR_CURVE = "markdown_what_is_pr_curve"

    @property
    def markdown_widget(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_PR_CURVE).format(
            self.vis_texts.definitions.f1_score
        )
        MarkdownWidget(name=self.MARKDOWN_PR_CURVE, title="Precision-Recall Curve", text=text)

    @property
    def chart_widget(self) -> ChartWidget:
        return ChartWidget(name="chart", figure=self.get_figure(), click_data=None)

    @property
    def collapsed_widget(self) -> CollapseWidget:
        text_pr_trade_offs = getattr(self.vis_texts, self.MARKDOWN_PR_TRADE_OFFS)
        text_pr_curve = getattr(self.vis_texts, self.MARKDOWN_WHAT_IS_PR_CURVE).format(
            self.vis_texts.definitions.confidence_score,
            self.vis_texts.definitions.true_positives,
            self.vis_texts.definitions.false_positives,
        )
        markdown_pr_trade_offs = MarkdownWidget(
            name=self.MARKDOWN_PR_TRADE_OFFS,
            title="About Trade-offs between precision and recall",
            text=text_pr_trade_offs,
        )
        markdown_whatis_pr_curve = MarkdownWidget(
            name=self.MARKDOWN_WHAT_IS_PR_CURVE,
            title="How the PR curve is built?",
            text=text_pr_curve,
        )
        return CollapseWidget(widgets=[markdown_pr_trade_offs, markdown_whatis_pr_curve])

    @property
    def notification_widget(self) -> NotificationWidget:
        desc = "".join(f"{ev.name}: {ev.mp.base_metrics()['mAP']:.2f}" for ev in self.eval_results)
        return NotificationWidget(name="map", title="mAP", desc=desc)

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "mAP (0.5:0.95)", "mAP (0.75)"]
        res["content"] = []
        for eval_result in self.eval_results:
            value_range = round(eval_result.mp.base_metrics()["mAP"], 2)
            value_75 = round(eval_result.mp.base_metrics()["AP75"], 2)
            model_name = eval_result.name
            row = [model_name, value_range, value_75]
            dct = {
                "row": row,
                "id": model_name,
                "items": row,
            }
            res["content"].append(dct)
        res["content"] = []
        for metric, value in self._loader.mp.metric_table().items():
            row = [metric, round(value, 2)]
            dct = {
                "row": row,
                "id": metric,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [
            {"customCell": True, "disableSort": True},
            {"disableSort": True},
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(data_source=res, show_header_controls=False, main_column=" ")

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        annotations_y = 0.92
        for eval_result in self.eval_results:
            # Precision-Recall curve
            pr_curve = eval_result.mp.pr_curve().copy()
            pr_curve[pr_curve == -1] = np.nan  # -1 is a placeholder for no GT
            pr_curve = np.nanmean(pr_curve, axis=-1)
            fig.add_trace(
                go.Scatter(
                    x=eval_result.mp.recThrs,
                    y=pr_curve,
                    mode="lines",
                    name=eval_result.name,
                    line=dict(color="#1f77b4"),
                    fill="tozeroy",
                    hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
                    showlegend=True,
                )
            )
            # fig.data[0].showlegend = True
            fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))
            fig.add_trace(
                go.Scatter(
                    x=eval_result.mp.recThrs,
                    y=[1] * len(eval_result.mp.recThrs),
                    name="Perfect",
                    line=dict(color="orange", dash="dash"),
                    showlegend=True,
                )
            )

            fig.add_annotation(
                text=f"mAP = {eval_result.mp.base_metrics()['mAP']:.2f}",
                xref="paper",
                yref="paper",
                x=0.98,
                y=annotations_y,
                showarrow=False,
                bgcolor="white",
            )
            annotations_y -= 0.05

        fig.update_layout(
            dragmode=False,
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
            width=600,
            height=500,
        )
        return fig
