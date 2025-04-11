from typing import List, Optional, Union

import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.model.model_api import Prediction
from supervisely.video.video import VideoFrameReader
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.video_annotation.video_object import VideoObject
from supervisely.video_annotation.video_object_collection import VideoObjectCollection


def apply_boxmot(
    tracker,
    predictions: Union[List[Prediction], List[Annotation]],
    class_names: List[str],
    frame_range: Optional[tuple] = None,
    video_path: Optional[str] = None,
    frames_count: Optional[int] = None,
) -> VideoAnnotation:
    results = []
    annotations = []
    frame_indexes = None
    if frame_range is not None:
        frame_indexes = list(range(frame_range[0], frame_range[1] + 1))
    if video_path is not None:
        frame_reader = VideoFrameReader(video_path, frame_indexes)
        frame_shape = frame_reader.frame_size()
        if frames_count is None:
            frames_count = frame_reader.frames_count()
        if frame_indexes is None:
            frame_indexes = list(range(frames_count))
        for prediction, frame in zip(predictions, frame_reader):
            if isinstance(prediction, Prediction):
                annotation = prediction.annotation
            else:
                annotation = prediction
            annotations.append(annotation)
            detections = to_boxes(annotation, class_names)  # N x (x, y, x, y, conf, label)
            tracks = tracker.update(
                detections, frame
            )  # M x (x, y, x, y, track_id, conf, label, det_id)
            results.append(tracks)
        return create_video_annotation(annotations, results, class_names, frame_shape, frames_count)
    else:
        if frame_indexes is None:
            frame_indexes = list(range(len(predictions)))
        for frame_index, prediction in zip(frame_indexes, predictions):
            frame = prediction.load_image()
            annotation = prediction.annotation
            annotations.append(annotation)
            detections = to_boxes(annotation, class_names)  # N x (x, y, x, y, conf, label)
            tracks = tracker.update(
                detections, frame
            )  # M x (x, y, x, y, track_id, conf, label, det_id)
            results.append(tracks)
        return create_video_annotation(
            annotations, results, class_names, list(frame.shape[:2]), frames_count
        )


def to_boxes(ann: Annotation, class_names: List[str]) -> np.ndarray:
    """
    Convert annotation to detections array in boxmot format.
    :param ann: Supervisely Annotation object
    :type ann: Annotation
    :param class_names: model class names
    :type class_names: List[str]
    :return: detections array N x (x, y, x, y, conf, label)
    :rtype: np.ndarray
    """
    # convert ann to N x (x, y, x, y, conf, cls) np.array
    cls2label = {class_name: i for i, class_name in enumerate(class_names)}
    detections = []
    for label in ann.labels:
        cat = cls2label[label.obj_class.name]
        bbox = label.geometry.to_bbox()
        conf = label.tags.get("confidence").value
        detections.append([bbox.left, bbox.top, bbox.right, bbox.bottom, conf, cat])
    detections = np.array(detections)
    return detections


def create_video_annotation(
    annotations: List[Annotation],
    tracking_results: list,
    class_names: List[str],
    frame_shape: tuple,
    frames_count: int,
) -> VideoAnnotation:
    img_h, img_w = frame_shape
    video_objects = {}  # track_id -> VideoObject
    frames = []
    cat2obj = {}
    name2cat = {class_name: i for i, class_name in enumerate(class_names)}
    obj_classes = {}
    for annotation in annotations:
        for label in annotation.labels:
            obj_classes.setdefault(label.obj_class.name, label.obj_class)
    for obj_name, cat in name2cat.items():
        obj_class = obj_classes.get(obj_name)
        if obj_class is None:
            raise ValueError(f"Object class {obj_name} not found in annotations.")
        cat2obj[cat] = obj_class
    for i, tracks in enumerate(tracking_results):
        frame_figures = []
        for track in tracks:
            # crop bbox to image size
            dims = np.array([img_w, img_h, img_w, img_h]) - 1
            track[:4] = np.clip(track[:4], 0, dims)
            x1, y1, x2, y2, track_id, conf, cat = track[:7]
            cat = int(cat)
            track_id = int(track_id)
            rect = Rectangle(y1, x1, y2, x2)
            video_object = video_objects.setdefault(track_id, VideoObject(cat2obj[cat]))
            frame_figures.append(VideoFigure(video_object, rect, i))
        frames.append(Frame(i, frame_figures))

    objects = list(video_objects.values())
    video_annotation = VideoAnnotation(
        img_size=frame_shape,
        frames_count=frames_count,
        objects=VideoObjectCollection(objects),
        frames=FrameCollection(frames),
    )
    return video_annotation
