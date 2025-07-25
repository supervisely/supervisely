
import argparse
import os
from contextlib import contextmanager
from typing import Dict, List, Union

import cv2
import numpy as np

from supervisely import (
    Annotation,
    Frame,
    FrameCollection,
    Label,
    Rectangle,
    VideoAnnotation,
    VideoFigure,
    VideoObject,
    VideoObjectCollection,
)
from supervisely.sly_logger import logger


class BaseDetection:
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like | NoneType
        A feature vector that describes the object contained in this image.
    sly_label : Label | NoneType
        A Supervisely Label object

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    sly_label : Label | NoneType
        A Supervisely Label object

    """

    def __init__(self, tlwh, confidence: float, feature=None, sly_label: Label = None):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self._sly_label = sly_label

    def __iter__(self):
        return iter([*self.tlwh, self.confidence, self.feature])

    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class BaseTrack:
    _count = 0

    track_id = 0
    is_activated = False
    state = "new"
    history = {}

    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count


    def activate(self, kalman_filter, frame_id):
        """
        Activate this track.
        Default implementation for BaseDetection objects.
        """
        # Initialize track-related attributes
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # For BaseDetection objects, convert tlwh to xywh for Kalman filter
        if hasattr(self, 'tlwh'):
            tlwh = self.tlwh
            xywh = [tlwh[0] + tlwh[2]/2, tlwh[1] + tlwh[3]/2, tlwh[2], tlwh[3]]
            self.mean, self.covariance = kalman_filter.initiate(xywh)
        
        self.state = TrackState.Tracked if hasattr(self, 'TrackState') else 1
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        # Set score from confidence if available
        if hasattr(self, 'confidence') and not hasattr(self, 'score'):
            self.score = self.confidence

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

    def get_sly_label(self):
        raise NotImplementedError


class BaseTracker:
    def __init__(self, settings=None):
        if settings is None:
            settings = {}

        default_settings = self.default_settings()
        default_settings.update(settings)

        self.args = argparse.Namespace(**default_settings)

    def default_settings(self):
        return {}

    @contextmanager
    def _video_frames_generator(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def frames_generator(self, source: str):
        if isinstance(source, str):

            def _gen():
                with self._video_frames_generator(source) as frames:
                    for frame in frames:
                        yield frame

            return _gen()
        elif isinstance(source, list) and isinstance(source[0], str):
            return [cv2.imread(img) for img in source]
        else:
            return source

    def track(
        self,
        source: Union[List[np.ndarray], List[str], str],
        frame_to_annotation: Dict[int, Annotation],
        pbar_cb=None,
    ):
        """To be overridden by subclasses."""
        raise NotImplementedError()

    def convert_annotation(self, annotation_for_frame: Annotation):
        """
        ИСПРАВЛЕННАЯ ВЕРСИЯ: Правильный порядок координат для tlwh формата
        tlwh = [left, top, width, height] = [x, y, w, h]
        """
        formatted_predictions = []
        sly_labels = []

        for label in annotation_for_frame.labels:
            confidence = 1.0
            if label.tags.get("confidence", None) is not None:
                confidence = label.tags.get("confidence").value
            elif label.tags.get("conf", None) is not None:
                confidence = label.tags.get("conf").value

            rectangle: Rectangle = label.geometry.to_bbox()
            
            # ИСПРАВЛЕНИЕ: Правильный порядок tlwh = [x, y, w, h] = [left, top, width, height]
            tlwh = [
                rectangle.left,    # x (left coordinate)
                rectangle.top,     # y (top coordinate)  
                rectangle.width,   # w (width)
                rectangle.height,  # h (height)
                confidence,
            ]

            formatted_predictions.append(tlwh)
            sly_labels.append(label)

        return formatted_predictions, sly_labels

    def update(
        self, img, annotation: Annotation, frame_index, tracks_data: Dict[int, List[Dict]] = None
    ):
        raise NotImplementedError()

    def correct_figure(self, img_size, figure):  # img_size — height, width tuple
        # check figure is within image bounds
        canvas_rect = Rectangle.from_size(img_size)
        if canvas_rect.contains(figure.to_bbox()) is False:
            # crop figure
            figures_after_crop = figure.crop(canvas_rect)
            if len(figures_after_crop) > 0:
                return figures_after_crop[0]
            else:
                return None
        else:
            return figure

    def update_track_data(self, tracks_data: dict, tracks: List[BaseTrack], frame_index: int):
        track_id_data = []
        labels_data = []

        for curr_track in tracks:
            track_id = curr_track.track_id

            if curr_track.get_sly_label() is not None:
                track_id_data.append(track_id)
                labels_data.append(curr_track.get_sly_label())

        tracks_data[frame_index] = {"ids": track_id_data, "labels": labels_data}

        return tracks_data

    def get_annotation(self, tracks_data: Dict, frame_shape, frames_count) -> VideoAnnotation:
        # Create and count object classes for each track
        object_classes = {}  # object_class_name -> object_class
        object_class_counter = {}  # track_id -> object_class_name -> count
        for frame_index, data in tracks_data.items():
            for track_id, label in zip(data["ids"], data["labels"]):
                label: Label
                object_classes.setdefault(label.obj_class.name, label.obj_class)
                object_class_counter.setdefault(track_id, {}).setdefault(label.obj_class.name, 0)
                object_class_counter[track_id][label.obj_class.name] += 1

        # Assign object classes to tracks
        track_obj_classes = {}  # track_id -> object_class
        for track_id, counters in object_class_counter.items():
            max_counter = -1
            obj_class_name = None
            for obj_class_name, count in counters.items():
                if count > max_counter:
                    max_counter = count
                    obj_class_name = obj_class_name
            track_obj_classes[track_id] = object_classes[obj_class_name]

        # Create video objects, figures and frames
        video_objects = {}  # track_id -> VideoObject
        frames = []
        for frame_index, data in tracks_data.items():
            frame_figures = []
            for track_id, label in zip(data["ids"], data["labels"]):
                label: Label
                video_object = video_objects.setdefault(
                    track_id, VideoObject(track_obj_classes[track_id])
                )
                frame_figures.append(VideoFigure(video_object, label.geometry, frame_index))
            frames.append(Frame(frame_index, frame_figures))

        objects = list(video_objects.values())
        return VideoAnnotation(
            img_size=frame_shape,
            frames_count=frames_count,
            objects=VideoObjectCollection(objects),
            frames=FrameCollection(frames),
        )