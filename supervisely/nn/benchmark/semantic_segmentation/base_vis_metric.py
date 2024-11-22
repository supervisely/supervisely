from typing import Dict, Optional

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)


class SemanticSegmVisMetric(BaseVisMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_result: SemanticSegmentationEvalResult

    def get_click_data(self) -> Optional[Dict]:
        if not self.clickable:
            return

        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}

        for key, v in self.eval_result.images_by_class.items():
            res["clickData"][key] = {}

            title = f"Class {key}: {len(v)} image{'s' if len(v) > 1 else ''}"
            res["clickData"][key]["title"] = title
            img_ids = [
                self.eval_result.matched_pair_data[gt_img_id].pred_image_info.id for gt_img_id in v
            ]
            res["clickData"][key]["imagesIds"] = img_ids

        return res

    def get_diff_data(self) -> Dict:
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
            key = click_data.setdefault(str(pred.id), {})
            key["imagesIds"] = [diff.id, gt.id, pred.id]
            key["title"] = f"Image: {pred.name}"
        return res
