from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ConfusionMatrix(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)

        self.clickable = True
        self.schema = Schema(
            self._loader.vis_texts,
            markdown_confusion_matrix=Widget.Markdown(title="Confusion Matrix", is_header=True),
            chart=Widget.Chart(),
        )

    def get_figure(self, widget: Widget.Chart):  # -> Optional[go.Figure]:
        import plotly.express as px  # pylint: disable=import-error

        confusion_matrix = self._loader.mp.confusion_matrix()
        # Confusion Matrix
        # TODO: Green-red
        cat_names = self._loader.mp.cat_names
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

    def get_click_data(self, widget: Widget.Chart) -> Optional[dict]:
        res = dict(projectMeta=self._loader.dt_project_meta.to_json())
        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for (pred_key, gt_key), matches_data in self._loader.click_data.confusion_matrix.items():
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
