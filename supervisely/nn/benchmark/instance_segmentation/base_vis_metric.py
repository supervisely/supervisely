from typing import Union

from supervisely.nn.benchmark.base_visualizer import BaseVisMetric
from supervisely.nn.benchmark.instance_segmentation.evaluator import (
    InstanceSegmentationEvalResult,
)
from supervisely.nn.benchmark.object_detection.evaluator import (
    ObjectDetectionEvalResult,
)


class DetectionVisMetric(BaseVisMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_result: Union[ObjectDetectionEvalResult, InstanceSegmentationEvalResult]
