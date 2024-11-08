from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class ConfusionMatrix(BaseVisMetric):
    MARKDOWN = "confusion_matrix"
    CHART = "confusion_matrix"

    def __init__(self, vis_texts, eval_result: ObjectDetectionEvalResult) -> None:
        super().__init__(vis_texts, [eval_result])
        self.eval_result = eval_result
        self.clickable = True

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_confusion_matrix
        return MarkdownWidget(self.MARKDOWN, "Confusion Matrix", text)

    @property
    def chart(self) -> ChartWidget:
        return ChartWidget(self.CHART, self._get_figure())

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
            res["clickData"][key]["title"] = f"Confusion Matrix. {gt_title} – {pred_title}"

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
