from typing import Dict, List, Tuple, Union

# pylint: disable=import-error
import clip
import numpy as np

from supervisely import Annotation, Label, VideoAnnotation
from supervisely.nn.tracker.deep_sort import generate_clip_detections as gdet
from supervisely.nn.tracker.deep_sort import preprocessing
from supervisely.nn.tracker.deep_sort.deep_sort import nn_matching
from supervisely.nn.tracker.deep_sort.deep_sort.detection import (
    Detection as dsDetection,
)
from supervisely.nn.tracker.deep_sort.deep_sort.track import Track as dsTrack
from supervisely.nn.tracker.deep_sort.deep_sort.track import TrackState
from supervisely.nn.tracker.deep_sort.deep_sort.tracker import Tracker as dsTracker
from supervisely.nn.tracker.tracker import BaseDetection, BaseTrack, BaseTracker
from supervisely.sly_logger import logger


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
        source: Union[List[np.ndarray], List[str], str],
        frame_to_annotation: Dict[int, Annotation],
        frame_shape: Tuple[int, int],
        pbar_cb=None,
    ) -> VideoAnnotation:
        """
        Track objects in the video using DeepSort algorithm.

        :param source: List of images, paths to images or path to the video file.
        :type source: List[np.ndarray] | List[str] | str
        :param frame_to_annotation: Dictionary with frame index as key and Annotation as value.
        :type frame_to_annotation: Dict[int, Annotation]
        :param frame_shape: Size of the frame (height, width).
        :type frame_shape: Tuple[int, int]
        :param pbar_cb: Callback to update progress bar.
        :type pbar_cb: Callable, optional

        :return: Video annotation with tracked objects.
        :rtype: VideoAnnotation

        :raises ValueError: If number of images and annotations are not the same.

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.nn.tracker import DeepSortTracker

            api = sly.Api()

            project_id = 12345
            video_id = 12345678
            video_path = "video.mp4"

            # Download video and get video info
            video_info = api.video.get_info_by_id(video_id)
            frame_shape = (video_info.frame_height, video_info.frame_width)
            api.video.download_path(id=video_id, path=video_path)

            # Run inference app to get detections
            task_id = 12345 # detection app task id
            session = sly.nn.inference.Session(api, task_id)
            annotations = session.inference_video_id(video_id, 0, video_info.frames_count)
            frame_to_annotation = {i: ann for i, ann in enumerate(annotations)}

            # Run tracker
            tracker = DeepSortTracker()
            video_ann = tracker.track(video_path, frame_to_annotation, frame_shape)

            # Upload result
            model_meta = session.get_model_meta()
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            project_meta = project_meta.merge(model_meta)
            api.project.update_meta(project_id, project_meta)
            api.video.annotation.append(video_id, video_ann)
        """
        if not isinstance(source, str):
            if len(source) != len(frame_to_annotation):
                raise ValueError("Number of images and annotations should be the same")

        tracks_data = {}
        logger.info("Starting deep_sort tracking with CLIP...")

        for frame_index, img in enumerate(self.frames_generator(source)):
            tracks_data = self.update(
                img, frame_to_annotation[frame_index], frame_index, tracks_data
            )

            if pbar_cb is not None:
                pbar_cb()

        tracks_data = self.clear_empty_ids(tracker_annotations=tracks_data)

        return self.get_annotation(
            tracks_data=tracks_data,
            frame_shape=frame_shape,
            frames_count=len(frame_to_annotation),
        )

    def update(
        self, img, annotation: Annotation, frame_index, tracks_data: Dict[int, List[Dict]] = None
    ):
        import torch

        detections = []
        try:
            pred, sly_labels = self.convert_annotation(annotation)
            det = torch.tensor(pred)

            # Process detections
            bboxes = det[:, :4].clone().cpu()
            # tlwh -> lthw
            bboxes = [bbox[[1, 0, 3, 2]] for bbox in bboxes]
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

        if tracks_data is None:
            tracks_data = {}
        self.update_track_data(
            tracks_data=tracks_data,
            tracks=[
                track
                for track in self.tracker.tracks
                if track.is_confirmed() or track.time_since_update <= 1
            ],
            frame_index=frame_index,
        )
        return tracks_data

    def update_track_data(self, tracks_data: dict, tracks: List[BaseTrack], frame_index: int):
        track_id_data = []
        labels_data = []

        for curr_track in tracks:
            track_id = curr_track.track_id - 1  # track_id starts from 1

            if curr_track.get_sly_label() is not None:
                track_id_data.append(track_id)
                labels_data.append(curr_track.get_sly_label())

        tracks_data[frame_index] = {"ids": track_id_data, "labels": labels_data}

        return tracks_data

    def clear_empty_ids(self, tracker_annotations):
        id_mappings = {}
        last_ordinal_id = 0

        for frame_index, data in tracker_annotations.items():
            data_ids_temp = []
            for current_id in data["ids"]:
                new_id = id_mappings.get(current_id, -1)
                if new_id == -1:
                    id_mappings[current_id] = last_ordinal_id
                    last_ordinal_id += 1
                    new_id = id_mappings.get(current_id, -1)
                data_ids_temp.append(new_id)
            data["ids"] = data_ids_temp

        return tracker_annotations
