from supervisely.nn.benchmark.base_visualizer import BaseVisMetrics
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class AveragePrecisionByClass(BaseVisMetrics):
    MARKDOWN_CLASS_AP = "markdown_class_ap_polar"
    MARKDOWN_CLASS_AP_BAR = "markdown_class_ap_bar"

    def get_figure(self):
        import plotly.graph_objects as go  # pylint: disable=import-error

        fig = go.Figure()
        labels = dict(r="Average Precision", theta="Class")
        cls_cnt = len(self.eval_results[0].mp.cat_names)
        for i, eval_result in enumerate(self.eval_results, 1):
            # AP per-class
            ap_per_class = eval_result.mp.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
            ap_per_class[ap_per_class == -1] = 0  # -1 is a placeholder for no GT

            trace_name = f"[{i}] {eval_result.name}"

            if cls_cnt >= 5:
                fig.add_trace(
                    go.Scatterpolar(
                        r=ap_per_class,
                        theta=eval_result.mp.cat_names,
                        name=trace_name,
                        marker=dict(color=eval_result.color),
                        hovertemplate=trace_name
                        + "<br>"
                        + labels["theta"]
                        + ": %{theta}<br>"
                        + labels["r"]
                        + ": %{r:.2f}<br>"
                        + "<extra></extra>",
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=eval_result.mp.cat_names,
                        y=ap_per_class,
                        name=trace_name,
                        width=0.2 if cls_cnt >= 5 else None,
                        marker=dict(color=eval_result.color, line=dict(width=0.7)),
                    )
                )

        if cls_cnt >= 5:
            fig.update_layout(
                width=800,
                height=800,
                margin=dict(l=80, r=80, t=0, b=0),
                modebar_add=["resetScale"],
                showlegend=True,
                polar=dict(radialaxis_range=[0, 1]),
            )
        else:
            fig.update_layout(
                xaxis_title="Class",
                yaxis_title="Average Precision",
                yaxis=dict(range=[0, 1.1]),
                barmode="group",
                width=700,
            )

        return fig

    @property
    def markdown_widget(self) -> MarkdownWidget:
        template_name = self.MARKDOWN_CLASS_AP
        if len(self.eval_results[0].mp.cat_names) < 5:
            template_name = self.MARKDOWN_CLASS_AP_BAR
        text: str = getattr(self.vis_texts, template_name).format(
            self.vis_texts.definitions.average_precision
        )
        return MarkdownWidget(
            name=self.MARKDOWN_CLASS_AP, title="Average Precision by Class", text=text
        )

    @property
    def chart_widget(self) -> ChartWidget:
        chart = ChartWidget(name="chart_class_ap", figure=self.get_figure())
        chart.set_click_data(
            gallery_id=self.explore_modal_table.id,
            click_data=self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].curveNumber}${'_'}${payload.points[0].theta}`,",
        )
        return chart

    def get_click_data(self):
        res = {}
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for i, eval_result in enumerate(self.eval_results):
            model_name = f"Model {i}"
            for cat_name, v in eval_result.click_data.objects_by_class.items():
                key = f"{i}_{cat_name}"
                ap_per_class_dict = res["clickData"].setdefault(key, {})

                img_ids = set()
                obj_ids = set()

                title = f"{model_name}, class: {len(v)} object{'s' if len(v) > 1 else ''}"
                ap_per_class_dict["title"] = title

                for x in v:
                    img_ids.add(x["dt_img_id"])
                    obj_ids.add(x["dt_obj_id"])

                ap_per_class_dict["imagesIds"] = list(img_ids)
                ap_per_class_dict["filters"] = [
                    {
                        "type": "tag",
                        "tagId": "confidence",
                        "value": [eval_result.mp.conf_threshold, 1],
                    },
                    {"type": "tag", "tagId": "outcome", "value": "TP"},
                    {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
                ]

        return res
