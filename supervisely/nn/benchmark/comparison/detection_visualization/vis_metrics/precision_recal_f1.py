from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    CollapseWidget,
    MarkdownWidget,
    NotificationWidget,
    TableWidget,
)


class PrecisionRecallF1(BaseVisMetrics):
    MARKDOWN = "markdown_PRF1"
    MARKDOWN_PRECISION_TITLE = "markdown_precision_per_class_title"
    MARKDOWN_RECALL_TITLE = "markdown_recall_per_class_title"
    MARKDOWN_F1_TITLE = "markdown_f1_per_class_title"

    @property
    def markdown_widget(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN).format(
            self.vis_texts.definitions.f1_score
        )
        return MarkdownWidget(name=self.MARKDOWN, title="Precision, Recall, F1-score", text=text)

    @property
    def precision_per_class_title_md(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_PRECISION_TITLE)
        text += self.vis_texts.clickable_label
        return MarkdownWidget(
            name="markdown_precision_per_class", title="Precision by Class", text=text
        )

    @property
    def recall_per_class_title_md(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_RECALL_TITLE)
        text += self.vis_texts.clickable_label
        return MarkdownWidget(name="markdown_recall_per_class", title="Recall by Class", text=text)

    @property
    def f1_per_class_title_md(self) -> MarkdownWidget:
        text: str = getattr(self.vis_texts, self.MARKDOWN_F1_TITLE)
        text += self.vis_texts.clickable_label
        return MarkdownWidget(name="markdown_f1_per_class", title="F1-score by Class", text=text)

    @property
    def chart_main_widget(self) -> ChartWidget:
        chart = ChartWidget(name="chart_PRF1", figure=self.get_main_figure())
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_click_data_main(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}`,",
        )
        return chart

    @property
    def chart_recall_per_class_widget(self) -> ChartWidget:
        chart = ChartWidget(
            name="chart_recall_per_class",
            figure=self.get_recall_per_class_figure(),
        )
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_per_class_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].label}`,",
        )
        return chart

    @property
    def chart_precision_per_class_widget(self) -> ChartWidget:
        chart = ChartWidget(
            name="chart_precision_per_class",
            figure=self.get_precision_per_class_figure(),
        )
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_per_class_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].label}`,",
        )
        return chart

    @property
    def chart_f1_per_class_widget(self) -> ChartWidget:
        chart = ChartWidget(name="chart_f1_per_class", figure=self.get_f1_per_class_figure())
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_per_class_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].label}`,",
        )
        return chart

    @property
    def table_widget(self) -> TableWidget:
        res = {}

        columns = [" ", "Precision", "Recall", "F1-score"]
        res["content"] = []
        for i, eval_result in enumerate(self.eval_results, 1):
            precision = round(eval_result.mp.json_metrics()["precision"], 2)
            recall = round(eval_result.mp.json_metrics()["recall"], 2)
            f1 = round(eval_result.mp.json_metrics()["f1"], 2)
            model_name = f"[{i}] {eval_result.name}"
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

        classes_cnt = len(self.eval_results[0].mp.cat_names)
        for i, eval_result in enumerate(self.eval_results, 1):
            precision = eval_result.mp.json_metrics()["precision"]
            recall = eval_result.mp.json_metrics()["recall"]
            f1 = eval_result.mp.json_metrics()["f1"]
            model_name = f"[{i}] {eval_result.name}"
            fig.add_trace(
                go.Bar(
                    x=["Precision", "Recall", "F1-score"],
                    y=[precision, recall, f1],
                    name=model_name,
                    width=0.2 if classes_cnt >= 5 else None,
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Value",
            yaxis=dict(range=[0, 1.1]),
            width=700,
        )

        return fig

    def get_recall_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        classes_cnt = len(self.eval_results[0].mp.cat_names)
        for i, eval_result in enumerate(self.eval_results, 1):
            model_name = f"[{i}] {eval_result.name}"
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["recall"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} Recall",
                    width=0.2 if classes_cnt >= 5 else None,
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            width=700 if classes_cnt < 5 else None,
        )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Recall", range=[0, 1])
        return fig

    def get_per_class_click_data(self):
        res = {}
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for i, eval_result in enumerate(self.eval_results):
            model_name = f"Model [{i + 1}] {eval_result.name}"
            for key, v in eval_result.click_data.objects_by_class.items():
                click_data = res["clickData"].setdefault(f"{i}_{key}", {})
                img_ids, obj_ids = set(), set()
                title = f"{model_name}. Class {key}: {len(v)} object{'s' if len(v) > 1 else ''}"
                click_data["title"] = title

                for x in v:
                    img_ids.add(x["dt_img_id"])
                    obj_ids.add(x["dt_obj_id"])

                click_data["imagesIds"] = list(img_ids)
                click_data["filters"] = [
                    {
                        "type": "tag",
                        "tagId": "confidence",
                        "value": [eval_result.mp.conf_threshold, 1],
                    },
                    {"type": "tag", "tagId": "outcome", "value": "TP"},
                    {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
                ]
        return res

    def get_precision_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        classes_cnt = len(self.eval_results[0].mp.cat_names)
        for i, eval_result in enumerate(self.eval_results, 1):
            model_name = f"[{i}] {eval_result.name}"
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["precision"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} Precision",
                    width=0.2 if classes_cnt >= 5 else None,
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            width=700 if classes_cnt < 5 else None,
        )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Precision", range=[0, 1])
        return fig

    def get_f1_per_class_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        classes_cnt = len(self.eval_results[0].mp.cat_names)
        for i, eval_result in enumerate(self.eval_results, 1):
            model_name = f"[{i}] {eval_result.name}"
            sorted_by_f1 = eval_result.mp.per_class_metrics().sort_values(by="f1")

            fig.add_trace(
                go.Bar(
                    y=sorted_by_f1["f1"],
                    x=sorted_by_f1["category"],
                    name=f"{model_name} F1-score",
                    width=0.2 if classes_cnt >= 5 else None,
                    marker=dict(color=eval_result.color, line=dict(width=0.7)),
                )
            )

        fig.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            width=700 if classes_cnt < 5 else None,
        )
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="F1-score", range=[0, 1])
        return fig

    def get_click_data_main(self):
        res = {}
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for i, eval_result in enumerate(self.eval_results):
            model_name = f"Model [{i + 1}] {eval_result.name}"
            click_data = res["clickData"].setdefault(i, {})
            img_ids, obj_ids = set(), set()
            objects_cnt = 0
            for outcome, matched_obj in eval_result.click_data.outcome_counts.items():
                if outcome == "TP":  # TODO: check if this is correct
                    objects_cnt += len(matched_obj)
                    for x in matched_obj:
                        img_ids.add(x["dt_img_id"])
                        obj_ids.add(x["dt_obj_id"])

            click_data["title"] = f"{model_name}, {objects_cnt} objects"
            click_data["imagesIds"] = list(img_ids)
            click_data["filters"] = [
                {
                    "type": "tag",
                    "tagId": "confidence",
                    "value": [eval_result.mp.conf_threshold, 1],
                },
                {"type": "tag", "tagId": "outcome", "value": "TP"},
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
