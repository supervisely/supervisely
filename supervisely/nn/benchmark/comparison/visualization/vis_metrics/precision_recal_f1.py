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


class PrecisionRecallF1(BaseVisMetric):
    MARKDOWN = "markdown_PR"

    @property
    def markdown_widget(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN).format(
            self.vis_texts.definitions.f1_score
        )
        return MarkdownWidget(name=self.MARKDOWN, title="Precision, Recall, F1-score", text=text)

    @property
    def chart_main_widget(self) -> ChartWidget:
        return ChartWidget(name="chart_pr_curve", figure=self.get_main_figure(), click_data=None)

    @property
    def chart_recall_per_class_widget(self) -> ChartWidget:
        return ChartWidget(
            name="chart_recall_per_class",
            figure=self.get_recall_per_class_figure(),
            click_data=None,
        )

    @property
    def chart_precision_per_class_widget(self) -> ChartWidget:
        return ChartWidget(
            name="chart_precision_per_class",
            figure=self.get_precision_per_class_figure(),
            click_data=None,
        )

    @property
    def chart_f1_per_class_widget(self) -> ChartWidget:
        return ChartWidget(
            name="chart_f1_per_class", figure=self.get_f1_per_class_figure(), click_data=None
        )

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "Precision", "Recall", "F1-score"]
        res["content"] = []
        for eval_result in self.eval_results:
            precision = round(eval_result.mp.json_metrics()["precision"], 2)
            recall = round(eval_result.mp.json_metrics()["recall"], 2)
            f1 = round(eval_result.mp.json_metrics()["f1"], 2)
            model_name = eval_result.name
            row = [model_name, precision, recall, f1]
            dct = {
                "row": row,
                "id": model_name,
                "items": row,
            }
            res["content"].append(dct)

        columns_options = [
            {"disableSort": True},
            {"disableSort": True},
            {"disableSort": True},
            {"disableSort": True},
        ]

        res["columns"] = columns
        res["columnsOptions"] = columns_options

        return TableWidget(
            name="table_precision_recall_f1",
            data=res,
            show_header_controls=False,
            # main_column=columns[0],
            fix_columns=1,
        )

    def get_main_figure(self):  # -> Optional[go.Figure]:
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()

        for eval_result in self.eval_results:
            precision = eval_result.mp.json_metrics()["precision"]
            recall = eval_result.mp.json_metrics()["recall"]
            f1 = eval_result.mp.json_metrics()["f1"]
            model_name = eval_result.name
            fig.add_trace(
                go.Bar(
                    x=["Precision", "Recall", "F1-score"],
                    y=[precision, recall, f1],
                    name=model_name,
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Value",
            yaxis=dict(range=[0, 1.1]),
        )

        return fig

    def get_recall_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        for eval_result in self.eval_results:
            model_name = eval_result.name
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["recall"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} Recall",
                )
            )

        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        return fig

    def get_precision_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        for eval_result in self.eval_results:
            model_name = eval_result.name
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["precision"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} Precision",
                )
            )

        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        return fig

    def get_f1_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        for eval_result in self.eval_results:
            model_name = eval_result.name
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["f1"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} F1-score",
                )
            )

        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Value", range=[0, 1])
        return fig
