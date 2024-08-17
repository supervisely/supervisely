from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisely.api.image_api import ImageInfo
from supervisely.nn.benchmark.visualization.vis_metric_base import MetricVis
from supervisely.nn.benchmark.visualization.vis_widgets import Schema, Widget

if TYPE_CHECKING:
    from supervisely.nn.benchmark.visualization.visualizer import Visualizer


class ModelPredictions(MetricVis):

    def __init__(self, loader: Visualizer) -> None:
        super().__init__(loader)
        self.schema = Schema(
            markdown_predictions_gallery=Widget.Markdown(
                title="Model Predictions", is_header=False
            ),
            markdown_predictions_table=Widget.Markdown(title="Prediction Table", is_header=True),
            # gallery=Widget.Gallery(is_table_gallery=True),
            table=Widget.Table(),
        )
        self._row_ids = None

    def get_table(self, widget: Widget.Table) -> dict:
        res = {}
        dt_project_id = self._loader.dt_project_info.id

        tmp = set()
        for dt_dataset in self._loader._api.dataset.get_list(dt_project_id):
            names = [x.name for x in self._loader._api.image.get_list(dt_dataset.id)]
            tmp.update(names)
        df = self._loader.mp.prediction_table().round(2)
        df = df[df["image_name"].isin(tmp)]
        columns_options = [{}] * len(df.columns)
        for idx, col in enumerate(columns_options):
            if idx == 0:
                continue
            columns_options[idx] = {"maxValue": df.iloc[:, idx].max()}
        table_model_preds = widget.table(df, columns_options=columns_options)
        tbl = table_model_preds.to_json()

        res["columns"] = tbl["columns"]
        res["columnsOptions"] = columns_options
        res["content"] = []

        key_mapping = {}
        for old, new in zip(ImageInfo._fields, self._loader._api.image.info_sequence()):
            key_mapping[old] = new

        self._row_ids = []

        for row in tbl["data"]["data"]:
            name = row["items"][0]
            info = self._loader.dt_images_dct_by_name[name]

            dct = {
                "row": {key_mapping[k]: v for k, v in info._asdict().items()},
                "id": info.name,
                "items": row["items"],
            }

            self._row_ids.append(dct["id"])
            res["content"].append(dct)

        return res

    def get_table_click_data(self, widget: Widget.Table) -> Optional[dict]:
        res = {}
        res["layoutTemplate"] = [
            {"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"},
            {"skipObjectTagsFiltering": ["outcome"], "columnTitle": "Prediction"},
            {"skipObjectTagsFiltering": False, "columnTitle": "Difference"},
        ]
        click_data = res.setdefault("clickData", {})

        default_filters = [
            {"type": "tag", "tagId": "confidence", "value": [0.6, 1]},
            {"type": "tag", "tagId": "outcome", "value": "FP"},
        ]

        l1 = list(self._loader.gt_images_dct.values())
        l2 = list(self._loader.dt_images_dct.values())
        l3 = list(self._loader.diff_images_dct.values())

        for gt, pred, diff in zip(l1, l2, l3):
            key = click_data.setdefault(str(pred.name), {})
            key["imagesIds"] = [gt.id, pred.id, diff.id]
            key["filters"] = default_filters

        return res
