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
        return MarkdownWidget(
            name=self.MARKDOWN_PR_CURVE, title="Precision-Recall Curve", text=text
        )

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
        desc = "".join(f"{ev.name}: {ev.mp.json_metrics()['mAP']:.2f}" for ev in self.eval_results)
        return NotificationWidget(name="map", title="mAP", desc=desc)

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "mAP (0.5:0.95)", "mAP (0.75)"]
        res["content"] = []
        for eval_result in self.eval_results:
            value_range = round(eval_result.mp.json_metrics()["mAP"], 2)
            value_75 = round(eval_result.mp.json_metrics()["AP75"], 2)
            model_name = eval_result.name
            row = [model_name, value_range, value_75]
            dct = {
                "row": row,
                "id": model_name,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [
            {"customCell": True, "disableSort": True},
            {"disableSort": True},
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(data=res, show_header_controls=False, main_column=" ")

    def get_figure(self):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        rec_thr = self.eval_results[0].mp.recThrs
        for eval_result in self.eval_results:
            pr_curve = eval_result.mp.pr_curve().copy()
            pr_curve[pr_curve == -1] = np.nan
            pr_curve = np.nanmean(pr_curve, axis=-1)

            line = go.Scatter(
                x=eval_result.mp.recThrs,
                y=pr_curve,
                mode="lines",
                name=eval_result.name,
                fill="tozeroy",
                hovertemplate=eval_result.name
                + "<br>Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
                showlegend=True,
            )
            fig.add_trace(line)

        fig.add_trace(
            go.Scatter(
                x=rec_thr,
                y=[1] * len(rec_thr),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
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
        )
        return fig
