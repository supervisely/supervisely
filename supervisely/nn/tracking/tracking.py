from typing import Any, Callable

from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.tracking.boxmot import apply_boxmot
from supervisely.video_annotation.video_annotation import VideoAnnotation


def _get_apply_fn(tracker: Any) -> Callable:
    if tracker.__class__.__module__.startswith("boxmot"):
        return apply_boxmot
    else:
        raise ValueError(
            f"Tracker {tracker.__class__.__module__} is not supported. Please, use boxmot tracker."
        )


def track(video_id: int, tracker, detector: ModelAPI, **kwargs) -> VideoAnnotation:
    apply_fn = _get_apply_fn(tracker)
    if "classes" in kwargs:
        classes = kwargs["classes"]
    else:
        classes = detector.get_classes()
    predictions = detector.predict_detached(video_id=video_id, **kwargs)
    return apply_fn(tracker, predictions, classes)