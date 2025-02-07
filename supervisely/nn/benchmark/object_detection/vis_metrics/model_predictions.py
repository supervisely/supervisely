from __future__ import annotations

from typing import Dict

from supervisely.api.image_api import ImageApi, ImageInfo
from supervisely.nn.benchmark.object_detection.base_vis_metric import DetectionVisMetric
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget, TableWidget


class ModelPredictions(DetectionVisMetric):
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
        tmp = set([d.pred_image_info.name for d in self.eval_result.matched_pair_data.values()])
        df = self.eval_result.mp.prediction_table().round(2)
        df = df[df["Image name"].isin(tmp)]
        columns_options = [
            {"maxWidth": "225px"},
            {"subtitle": "objects count", "tooltip": "Number of ground truth objects on the image"},
            {"subtitle": "objects count", "tooltip": "Number of predicted objects on the image"},
            {
                "subtitle": "objects count",
                "tooltip": "True Positive objects count (correctly detected)",
            },
            {
                "subtitle": "objects count",
                "tooltip": "False Positive objects count (incorrect detections)",
            },
            {
                "subtitle": "objects count",
                "tooltip": "False Negative objects count (missed detections)",
            },
            {"maxValue": 1, "tooltip": "Precision (positive predictive value)"},
            {"maxValue": 1, "tooltip": "Recall (sensitivity)"},
            {"maxValue": 1, "tooltip": "F1 score (harmonic mean of precision and recall)"},
        ]

        columns = df.columns.tolist()[1:]  # exclude sly_id
        content = []

        key_mapping = {}
        for old, new in zip(ImageInfo._fields, ImageApi.info_sequence()):
            key_mapping[old] = new

        self._row_ids = []
        df = df.replace({float("nan"): None}) # replace NaN / float("nan") with None

        for row in df.values.tolist():
            sly_id = row.pop(0)
            info = self.eval_result.matched_pair_data[sly_id].gt_image_info

            dct = {
                "row": {key_mapping[k]: v for k, v in info._asdict().items()},
                "id": info.name,
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
            {"skipObjectTagsFiltering": True, "columnTitle": "Ground Truth"},
            {"skipObjectTagsFiltering": ["outcome"], "columnTitle": "Prediction"},
            {"skipObjectTagsFiltering": ["confidence"], "columnTitle": "Difference"},
        ]
        click_data = res.setdefault("clickData", {})

        default_filters = [
            {
                "type": "tag",
                "tagId": "confidence",
                "value": [self.eval_result.mp.conf_threshold, 1],
            },
            # {"type": "tag", "tagId": "outcome", "value": "FP"},
        ]

        for pairs_data in self.eval_result.matched_pair_data.values():
            gt = pairs_data.gt_image_info
            pred = pairs_data.pred_image_info
            diff = pairs_data.diff_image_info
            assert gt.name == pred.name == diff.name
            key = click_data.setdefault(str(pred.name), {})
            key["imagesIds"] = [gt.id, pred.id, diff.id]
            key["filters"] = default_filters
            key["title"] = f"Image: {pred.name}"
            image_id = pred.id
            ann_json = pairs_data.pred_annotation.to_json()
            assert image_id == pred.id
            object_bindings = []
            for obj in ann_json["objects"]:
                for tag in obj["tags"]:
                    if tag["name"] == "matched_gt_id":
                        object_bindings.append(
                            [
                                {
                                    "id": obj["id"],
                                    "annotationKey": image_id,
                                },
                                {
                                    "id": int(tag["value"]),
                                    "annotationKey": gt.id,
                                },
                            ]
                        )

            image_id = diff.id
            ann_json = pairs_data.diff_annotation.to_json()
            assert image_id == diff.id
            for obj in ann_json["objects"]:
                for tag in obj["tags"]:
                    if tag["name"] == "matched_gt_id":
                        object_bindings.append(
                            [
                                {
                                    "id": obj["id"],
                                    "annotationKey": image_id,
                                },
                                {
                                    "id": int(tag["value"]),
                                    "annotationKey": pred.id,
                                },
                            ]
                        )
            key["objectsBindings"] = object_bindings

        return res
