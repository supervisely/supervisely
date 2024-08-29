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
            width=1000,
            height=1000,
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

        for k, val in self._loader.click_data.confusion_matrix.items():
            pred_key, gt_key = k
            key = gt_key + self._keypair_sep + pred_key
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []
            gt_title = f"GT: '{gt_key}'" if gt_key != "(None)" else "No GT Objects"
            pred_title = f"Predicted: '{pred_key}'" if pred_key != "(None)" else "No Predictions"
            res["clickData"][key]["title"] = f"Confusion Matrix. {gt_title} – {pred_title}"

            img_ids = set()
            obj_ids = set()
            for x in val:
                if x["dt_obj_id"] is not None:
                    image_id = self._loader.dt_images_dct[x["dt_img_id"]].id
                    img_ids.add(image_id)
                    obj_ids.add(x["dt_obj_id"])
                else:
                    img_ids.add(self._loader.gt_images_dct[x["gt_img_id"]].id)
                    obj_ids.add(x["gt_obj_id"])

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
