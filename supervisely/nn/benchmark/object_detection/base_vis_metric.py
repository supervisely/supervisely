from typing import Dict, Optional

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)


class DetectionVisMetric(BaseVisMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_result: ObjectDetectionEvalResult

    def get_click_data(self) -> Optional[Dict]:
        if not self.clickable:
            return

        res = {}

        res["layoutTemplate"] = [None, None, None]
        res["clickData"] = {}
        for key, v in self.eval_result.click_data.objects_by_class.items():
            res["clickData"][key] = {}
            res["clickData"][key]["imagesIds"] = []

            img_ids = set()
            obj_ids = set()

            res["clickData"][key][
                "title"
            ] = f"{key} class: {len(v)} object{'s' if len(v) > 1 else ''}"

            for x in v:
                img_ids.add(x["dt_img_id"])
                obj_id = x["dt_obj_id"]
                if obj_id is not None:
                    obj_ids.add(obj_id)

            res["clickData"][key]["imagesIds"] = list(img_ids)
            res["clickData"][key]["filters"] = [
                {
                    "type": "tag",
                    "tagId": "confidence",
                    "value": [self.eval_result.mp.conf_threshold, 1],
                },
                {"type": "tag", "tagId": "outcome", "value": "TP"},
                {"type": "specific_objects", "tagId": None, "value": list(obj_ids)},
            ]

        return res
