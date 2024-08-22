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
            labels=dict(x="Ground Truth", y="Predicted", color="Count"),
            # title="Confusion Matrix (log-scale)",
            width=1000,
            height=1000,
        )

        # Hover text
        fig.update_traces(
            customdata=confusion_matrix,
            hovertemplate="Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}",
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

        unique_pairs = set()
        filtered_pairs = []
        for k, val in self._loader.click_data.confusion_matrix.items():
            ordered_pair = tuple(sorted(k))
            if ordered_pair not in unique_pairs:
                unique_pairs.add(ordered_pair)
            else:
                continue

            subkey1, subkey2 = ordered_pair
            key = subkey1 + self._keypair_sep + subkey2
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []
            res["clickData"][key]["title"] = f"Class Pair: {subkey1} - {subkey2}"
            sub_title_1 = f"{subkey1} (GT)" if subkey1 != "(None)" else "No GT"
            sub_title_2 = f"{subkey2} (Prediction)" if subkey2 != "(None)" else "No Prediction"
            res["clickData"][key]["title"] = f"Confusion Matrix: {sub_title_1} – {sub_title_2}"

            img_ids = set()
            obj_ids = set()
            for x in val:
                image_id = self._loader.dt_images_dct[x["dt_img_id"]].id
                img_ids.add(image_id)
                obj_ids.add(x["dt_obj_id"])

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
