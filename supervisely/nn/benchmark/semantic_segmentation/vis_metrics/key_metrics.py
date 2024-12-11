from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class KeyMetrics(SemanticSegmVisMetric):

    @property
    def md(self) -> MarkdownWidget:
        return MarkdownWidget(
            "markdown_header",
            "Key Metrics",
            text=self.vis_texts.markdown_key_metrics,
        )

    @property
    def table(self) -> TableWidget:
        columns = ["metrics", "values"]
        content = []

        metrics = self.eval_result.mp.key_metrics().copy()
        metrics["mPixel accuracy"] = round(metrics["mPixel accuracy"] * 100, 2)

        for metric, value in metrics.items():
            row = [metric, round(value, 2)]
            dct = {"row": row, "id": metric, "items": row}
            content.append(dct)

        columns_options = [{"disableSort": True}, {"disableSort": True}]
        data = {"columns": columns, "columnsOptions": columns_options, "content": content}

        table = TableWidget(
            name="table_key_metrics",
            data=data,
            fix_columns=1,
            width="60%",
            show_header_controls=False,
            main_column=columns[0],
        )
        return table

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget("base_metrics_chart", self.get_figure())

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        metrics = self.eval_result.mp.key_metrics().copy()
        metrics["mPixel accuracy"] = round(metrics["mPixel accuracy"] * 100, 2)
        fig.add_trace(
            go.Scatterpolar(
                r=list(metrics.values()) + [list(metrics.values())[0]],
                theta=list(metrics.keys()) + [list(metrics.keys())[0]],
                # fill="toself",
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
            ),
        )
        return fig
