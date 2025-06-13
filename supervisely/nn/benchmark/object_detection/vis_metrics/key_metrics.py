from __future__ import annotations

from typing import Dict

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import (
    ChartWidget,
    MarkdownWidget,
    TableWidget,
)


class KeyMetrics(DetectionVisMetric):
    MARKDOWN = "key_metrics"
    CHART = "key_metrics"
    TABLE = "key_metrics"

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_key_metrics.format(
            self.vis_texts.definitions.iou_threshold,
            self.vis_texts.definitions.average_precision,
            self.vis_texts.definitions.confidence_score,
        )
        return MarkdownWidget(self.MARKDOWN, "Key Metrics", text)

    @property
    def table(self) -> TableWidget:
        columns = ["metrics", "values"]
        content = []
        for metric, value in self.eval_result.mp.metric_table().items():
            if metric == "AP_custom":
                metric += "*"
            row = [metric, round(value, 2)]
            dct = {
                "row": row,
                "id": metric,
                "items": row,
            }
            content.append(dct)

        columns_options = [
            {"disableSort": True},  # , "ustomCell": True},
            {"disableSort": True},
        ]

        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        table = TableWidget(
            name=self.TABLE,
            data=data,
            fix_columns=1,
            width="60%",
            show_header_controls=False,
            main_column=columns[0],
            page_size=15,
        )
        return table

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

    def _get_figure(self):  # -> go.Figure:
        import plotly.graph_objects as go  # pylint: disable=import-error

        # Overall Metrics
        base_metrics = self.eval_result.mp.base_metrics()
        r = list(base_metrics.values())
        theta = [self.eval_result.mp.metric_names[k] for k in base_metrics.keys()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                # fill="toself",
                name="Overall Metrics",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0.0, 1.0],
                    ticks="outside",
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            dragmode=False,
            margin=dict(l=25, r=25, t=25, b=25),
        )
        fig.update_layout(
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
            )
        )
        return fig

    def get_click_data(self) -> Dict:
        if not self.clickable:
            return
        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for outcome, matches_data in self.eval_result.click_data.outcome_counts.items():
            res["clickData"][outcome] = {}
            res["clickData"][outcome]["imagesIds"] = []

            img_ids = set()
            for match_data in matches_data:
                pairs_data = self.eval_result.matched_pair_data[match_data["gt_img_id"]]
                if outcome == "FN":
                    img_ids.add(pairs_data.diff_image_info.id)
                else:
                    img_ids.add(pairs_data.pred_image_info.id)

            res["clickData"][outcome][
                "title"
            ] = f"{outcome}: {len(matches_data)} object{'s' if len(matches_data) > 1 else ''}"
            res["clickData"][outcome]["imagesIds"] = list(img_ids)
            res["clickData"][outcome]["filters"] = [
                {"type": "tag", "tagId": "confidence", "value": [0, 1]},
                {"type": "tag", "tagId": "outcome", "value": outcome},
            ]

        return res

    @property
    def custom_ap_description_md(self) -> MarkdownWidget:
        if not self.eval_result.different_iou_thresholds_per_class:
            return None
        return MarkdownWidget(
            "custom_ap_description",
            "Custom AP per Class",
            self.vis_texts.markdown_AP_custom_description,
        )
