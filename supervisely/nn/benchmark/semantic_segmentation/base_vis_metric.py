from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.semantic_segmentation.evaluator import (
    SemanticSegmentationEvalResult,
)


class SemanticSegmVisMetric(BaseVisMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_result: SemanticSegmentationEvalResult
