from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)


class DetectionVisMetric(BaseVisMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_result: ObjectDetectionEvalResult
