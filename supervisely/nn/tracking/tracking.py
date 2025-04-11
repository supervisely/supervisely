from typing import Any, List, Union

from supervisely.annotation.annotation import Annotation
from supervisely.nn.model_api import Prediction
from supervisely.nn.tracking.boxmot import apply_boxmot
from supervisely.video_annotation.video_annotation import VideoAnnotation


def define_tracker(tracker: Any) -> str:
    if tracker.__class__.__module__.startswith("boxmot"):
        return "boxmot"
    raise NotImplementedError(f"Tracker {tracker.__class__.__module__} is not supported. ")


def track(tracker: Any, predictions: Union[List[Prediction], List[Annotation]]) -> VideoAnnotation:
    tracker_str = define_tracker(tracker)
    if tracker_str == "boxmot":
        return apply_boxmot(tracker, predictions)
