from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ConfusionMatrix(DetectionVisMetric):
    MARKDOWN = "confusion_matrix"
    CHART = "confusion_matrix"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True
        self._keypair_sep: str = "-"

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_confusion_matrix
        return MarkdownWidget(self.MARKDOWN, "Confusion Matrix", text)

    @property
    def chart(self) -> ChartWidget:
        chart = ChartWidget(self.CHART, self._get_figure())
        chart.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
            chart_click_extra="'getKey': (payload) => `${payload.points[0].x}${'-'}${payload.points[0].y}`, 'keySeparator': '-',",
        )
        return chart

    def _get_figure(self):  # -> go.Figure:
        import plotly.express as px  # pylint: disable=import-error

        confusion_matrix = self.eval_result.mp.confusion_matrix()
        # TODO: Green-red
        cat_names = self.eval_result.mp.cat_names
        none_name = "(None)"

        with np.errstate(divide="ignore"):
            loged_cm = np.log(confusion_matrix)

        df = pd.DataFrame(
            loged_cm,
            index=cat_names + [none_name],
            columns=cat_names + [none_name],
        )
        fig = px.imshow(
            df,
            labels=dict(x="Ground Truth", y="Predicted", color="Objects Count"),
            # title="Confusion Matrix (log-scale)",
            width=1000 if len(cat_names) > 10 else 600,
            height=1000 if len(cat_names) > 10 else 600,
        )

        # Hover text
        fig.update_traces(
            customdata=confusion_matrix,
            hovertemplate="Objects Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}",
            colorscale="Viridis",
        )

        # Text on cells
        if len(cat_names) <= 20:
            fig.update_traces(text=confusion_matrix, texttemplate="%{text}")

        # fig.show()
        return fig

    def get_click_data(self) -> Dict:
        res = dict(projectMeta=self.eval_result.pred_project_meta.to_json())
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for (
            pred_key,
            gt_key,
        ), matches_data in self.eval_result.click_data.confusion_matrix.items():
            key = gt_key + self._keypair_sep + pred_key
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []
            gt_title = f"GT: '{gt_key}'" if gt_key != "(None)" else "No GT Objects"
            pred_title = f"Predicted: '{pred_key}'" if pred_key != "(None)" else "No Predictions"
            res["clickData"][key]["title"] = f"Confusion Matrix. {gt_title} ― {pred_title}"

            img_ids = set()
            obj_ids = set()
            for match_data in matches_data:
                if match_data["dt_obj_id"] is not None:
                    img_ids.add(match_data["dt_img_id"])
                    obj_ids.add(match_data["dt_obj_id"])
                else:
                    img_ids.add(match_data["gt_img_id"])
                    obj_ids.add(match_data["gt_obj_id"])

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
