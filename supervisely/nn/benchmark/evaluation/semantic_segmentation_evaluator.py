from supervisely.nn.benchmark.evaluation import BaseEvaluator
import os
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.functions import (
    evaluate_segmentation_quality,
)


class SemanticSegmentationEvaluator(BaseEvaluator):
    def __init__(
        self,
        api,
        gt_project_id: int,
        pred_project_id: int,
        result_dir: str = "./evaluation",
    ):
        self.api = api
        self.gt_project_id = gt_project_id
        self.pred_project_id = pred_project_id
        self.result_dir = result_dir

    def evaluate(self):
        evaluate_segmentation_quality(
            api=self.api,
            gt_project_id=self.gt_project_id,
            pred_project_id=self.pred_project_id,
            subset_size=3,
            n_iter=15,
            batch_size=8,
            result_dir=self.result_dir,
        )
