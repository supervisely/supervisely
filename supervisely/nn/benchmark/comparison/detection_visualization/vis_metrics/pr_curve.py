import numpy as np

from supervisely.imaging.color import hex2rgb
from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
    TableWidget,
)


class PrCurve(BaseVisMetrics):
    MARKDOWN_PR_CURVE = "markdown_pr_curve"
    MARKDOWN_PR_TRADE_OFFS = "markdown_trade_offs"
    MARKDOWN_WHAT_IS_PR_CURVE = "markdown_what_is_pr_curve"

    @property
    def markdown_widget(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_PR_CURVE).format(
            self.vis_texts.definitions.about_pr_tradeoffs
        )
        return MarkdownWidget(
            name=self.MARKDOWN_PR_CURVE, title="mAP & Precision-Recall Curve", text=text
        )

    @property
    def chart_widget(self) -> ChartWidget:
        return ChartWidget(name="chart_pr_curve", figure=self.get_figure())

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
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "mAP (0.5:0.95)", "mAP (0.75)"]
        res["content"] = []
        for i, eval_result in enumerate(self.eval_results, 1):
            value_range = round(eval_result.mp.json_metrics()["mAP"], 2)
            value_75 = eval_result.mp.json_metrics()["AP75"] or "-"
            value_75 = round(value_75, 2) if isinstance(value_75, float) else value_75
            model_name = f"[{i}] {eval_result.name}"
            row = [model_name, value_range, value_75]
            dct = {
                "row": row,
                "id": model_name,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [
            {"disableSort": True},
            {"disableSort": True},
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(
            name="table_pr_curve",
            data=res,
            show_header_controls=False,
            # main_column=columns[0],
            fix_columns=1,
        )

    def get_figure(self):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        rec_thr = self.eval_results[0].mp.recThrs
        for i, eval_result in enumerate(self.eval_results, 1):
            pr_curve = eval_result.mp.pr_curve().copy()
            pr_curve[pr_curve == -1] = np.nan
            pr_curve = np.nanmean(pr_curve, axis=-1)

            name = f"[{i}] {eval_result.name}"
            color = ",".join(map(str, hex2rgb(eval_result.color))) + ",0.1"
            line = go.Scatter(
                x=eval_result.mp.recThrs,
                y=pr_curve,
                mode="lines",
                name=name,
                fill="tozeroy",
                fillcolor=f"rgba({color})",
                line=dict(color=eval_result.color),
                hovertemplate=name + "<br>Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
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
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        return fig
