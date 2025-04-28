from typing import Any, List, Union

from supervisely.annotation.annotation import Annotation
from supervisely.nn.model.model_api import ModelAPI, Prediction
from supervisely.nn.tracking.boxmot import apply_boxmot
from supervisely.video_annotation.video_annotation import VideoAnnotation


def define_tracker(tracker: Any) -> str:
    if tracker.__class__.__module__.startswith("boxmot"):
        return "boxmot"
    raise NotImplementedError(f"Tracker {tracker.__class__.__module__} is not supported. ")


def track(
    tracker: Any, predictions: Union[List[Prediction], List[Annotation]], classes: List[str]
) -> VideoAnnotation:
    tracker_str = define_tracker(tracker)
    if tracker_str == "boxmot":
        return apply_boxmot(tracker, predictions, classes)
    raise NotImplementedError(
        f"Tracker {tracker.__class__.__module__} is not supported. "
        f"Please implement a new tracker or use the existing one."
    )


def track_predictor(video_id: int, tracker, detector: ModelAPI, **kwargs) -> VideoAnnotation:
    tracker_str = define_tracker(tracker)
    if tracker_str == "boxmot":
        if "classes" in kwargs:
            classes = kwargs["classes"]
        else:
            classes = detector.get_classes()
        predictions = detector.predict_detached(video_id=video_id, **kwargs)
        return apply_boxmot(tracker, predictions, classes)
    raise NotImplementedError(
        f"Tracker {tracker.__class__.__module__} is not supported. "
        f"Please implement a new tracker or use the existing one."
    )
