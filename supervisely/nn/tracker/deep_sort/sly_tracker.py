from typing import Dict, List, Union

# pylint: disable=import-error
import clip
import cv2
import numpy as np

from supervisely import Annotation, Label
from supervisely.nn.tracker.ann_keeper import AnnotationKeeper
from supervisely.nn.tracker.tracker import BaseDetection, BaseTrack, BaseTracker
from supervisely.sly_logger import logger

from . import generate_clip_detections as gdet
from . import preprocessing
from .deep_sort import nn_matching
from .deep_sort.detection import Detection as dsDetection
from .deep_sort.track import Track as dsTrack
from .deep_sort.track import TrackState
from .deep_sort.tracker import Tracker as dsTracker


class Detection(BaseDetection, dsDetection):
    def __init__(self, sly_label: Label, tlwh, confidence, feature):
        dsDetection.__init__(self, tlwh, confidence, feature)
        self.sly_label = sly_label

    def get_sly_label(self):
        return self.sly_label


class Track(BaseTrack, dsTrack):
    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        detection: Detection = None,
    ):
        dsTrack.__init__(self, mean, covariance, track_id, n_init, max_age, feature=None)

        self.state = TrackState.Confirmed
        self.features = []
        self._sly_label = None
        self.class_num = None
        if detection is not None:
            self.features.append(detection.feature)
            self._sly_label = detection.get_sly_label()
            self.class_num = self._sly_label.obj_class.name

    def get_sly_label(self):
        return self._sly_label

    def clean_sly_label(self):
        self._sly_label = None

    def update(self, kf, detection: Detection):
        dsTrack.update(self, kf, detection)
        self._sly_label = detection.get_sly_label()


class _dsTracker(dsTracker):
    """Extend deep sort tracker to support Supervisely labels."""

    def update(self, detections: List[Detection]):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Clean up all previous labels
        for track in self.tracks:
            track: Track
            track.clean_sly_label()

        dsTracker.update(self, detections)

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection)
        )
        self._next_id += 1


class DeepSortTracker(BaseTracker):
    def __init__(self, settings: Dict = None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        model_filename = "ViT-B/32"  # initialize deep sort
        logger.info("Loading CLIP...")
        model, transform = clip.load(model_filename, device=self.device)
        self.encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=self.device)
        metric = nn_matching.NearestNeighborDistanceMetric(  # calculate cosine distance metric
            "cosine", self.args.max_cosine_distance, self.args.nn_budget
        )
        self.tracker = _dsTracker(metric, n_init=1)

    def default_settings(self):
        """To be overridden by subclasses."""
        return {"nms_max_overlap": 1.0, "max_cosine_distance": 0.6, "nn_budget": None}

    def track(
        self,
        images: Union[List[np.ndarray], List[str]],
        frame_to_annotation: Dict[int, Annotation],
        annotation_keeper: AnnotationKeeper = None,
        pbar_cb=None,
    ) -> Dict[int, Dict]:
        if len(images) != len(frame_to_annotation):
            raise ValueError("Number of images and annotations should be the same")

        import torch

        tracks_data = {}
        logger.info("Starting deep_sort tracking with CLIP...")
        # frame_index = 0
        for image_index, frame_index in enumerate(frame_to_annotation.keys()):
            detections = []
            img = images[image_index]
            if isinstance(img, str):
                img = cv2.imread(img)

            try:
                pred, sly_labels = self.convert_annotation(frame_to_annotation[frame_index])
                det = torch.tensor(pred)

                # Process detections
                bboxes = det[:, :4].clone().cpu()
                confs = det[:, 4]

                # encode yolo detections and feed to tracker
                features = self.encoder(img, bboxes)
                detections = [
                    Detection(sly_label, bbox, conf, feature)
                    for bbox, conf, feature, sly_label in zip(bboxes, confs, features, sly_labels)
                ]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.sly_label.obj_class.name for d in detections])
                indices_of_alive_labels = preprocessing.non_max_suppression(
                    boxs, class_nums, self.args.nms_max_overlap, scores
                )
                detections = [detections[i] for i in indices_of_alive_labels]
            except Exception as ex:
                import traceback

                logger.info(f"frame {frame_index} skipped on tracking")
                logger.debug(traceback.format_exc())

            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            self.update_track_data(
                tracks_data=tracks_data,
                tracks=[
                    track
                    for track in self.tracker.tracks
                    if track.is_confirmed() or track.time_since_update <= 1
                ],
                frame_index=frame_index,
                img_size=img.shape[:2],
            )

            if pbar_cb is not None:
                pbar_cb()

        tracks_data = self.clear_empty_ids(tracker_annotations=tracks_data)

        if annotation_keeper is not None:
            annotation_keeper.add_figures_by_frames(tracks_data)
        return tracks_data
