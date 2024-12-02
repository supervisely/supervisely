from __future__ import annotations

from typing import Dict

from supervisely.api.image_api import ImageApi, ImageInfo
from supervisely.nn.benchmark.semantic_segmentation.base_vis_metric import (
    SemanticSegmVisMetric,
)
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget, TableWidget


class ModelPredictions(SemanticSegmVisMetric):
    MARKDOWN = "model_predictions"
    TABLE = "model_predictions"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clickable = True
        self._row_ids = None  #  TODO: check if this is used

    @property
    def md(self) -> MarkdownWidget:
        text = self.vis_texts.markdown_predictions_table
        return MarkdownWidget(self.MARKDOWN, "Prediction details for every image", text)

    @property
    def table(self) -> TableWidget:
        df = self.eval_result.mp.metric_table().round(2)

        # tmp = set([d.pred_image_info.name for d in self.eval_result.matched_pair_data.values()])
        # df = df[df["Image name"].isin(tmp)]
        columns_options = [
            {"maxWidth": "225px"},
            {"maxValue": 1, "tooltip": "Pixel accuracy"},
            {"maxValue": 1, "tooltip": "Precision (positive predictive value)"},
            {"maxValue": 1, "tooltip": "Recall (sensitivity)"},
            {"maxValue": 1, "tooltip": "F1 score (harmonic mean of precision and recall)"},
            {"maxValue": 1, "tooltip": "IoU (Intersection over Union)"},
            {"maxValue": 1, "tooltip": "Boundary IoU"},
            {"maxValue": 1, "tooltip": "Boundary EoU"},
            {"maxValue": 1, "tooltip": "Extent EoU"},
            {"maxValue": 1, "tooltip": "Segment EoU"},
            {"maxValue": 1, "tooltip": "Boundary EoU renormed"},
            {"maxValue": 1, "tooltip": "Extent EoU renormed"},
            {"maxValue": 1, "tooltip": "Segment EoU renormed"},
        ]

        # columns = df.columns.tolist()[1:]  # exclude sly_id
        columns = df.columns.tolist()
        content = []

        # key_mapping = {}
        # for old, new in zip(ImageInfo._fields, ImageApi.info_sequence()):
        #     key_mapping[old] = new

        self._row_ids = []
        df = df.replace({float("nan"): None})  # replace NaN / float("nan") with None

        for row in df.values.tolist():
            # sly_id = row.pop(0)
            # info = self.eval_result.matched_pair_data[sly_id].gt_image_info
            name = row[0]

            dct = {
                "row": row,
                "id": name,
                "items": row,
            }

            self._row_ids.append(dct["id"])
            content.append(dct)

        data = {
            "columns": columns,
            "columnsOptions": columns_options,
            "content": content,
        }
        table = TableWidget(
            name=self.TABLE,
            data=data,
            fix_columns=1,
        )
        table.set_click_data(
            self.explore_modal_table.id,
            self.get_click_data(),
        )
        return table

    def get_click_data(self) -> Dict:
        res = {}
        res["layoutTemplate"] = [
            {"columnTitle": "Original Image"},
            {"columnTitle": "Ground Truth Masks"},
            {"columnTitle": "Predicted Masks"},
        ]
        click_data = res.setdefault("clickData", {})

        for pairs_data in self.eval_result.matched_pair_data.values():
            gt = pairs_data.gt_image_info
            pred = pairs_data.pred_image_info
            diff = pairs_data.diff_image_info
            assert gt.name == pred.name == diff.name
            key = click_data.setdefault(str(pred.name), {})
            key["imagesIds"] = [diff.id, gt.id, pred.id]
            key["title"] = f"Image: {pred.name}"

        return res
