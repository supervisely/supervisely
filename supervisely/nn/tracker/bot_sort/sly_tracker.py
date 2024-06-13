from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import requests

from supervisely import Annotation, Label
from supervisely.nn.tracker.tracker import BaseDetection as Detection
from supervisely.nn.tracker.tracker import BaseTrack, BaseTracker
from supervisely.sly_logger import logger

from . import matching
from .bot_sort import (
    BoTSORT,
    STrack,
    TrackState,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)


class Track(BaseTrack, STrack):
    def __init__(self, tlwh, confidence: float, feature=None, sly_label: Label = None):
        self.track_id = None
        STrack.__init__(self, tlwh, confidence, sly_label.obj_class.name, feature)
        self.sly_label = sly_label

    def get_sly_label(self):
        return self.sly_label

    def update(self, new: Track, frame_id):
        STrack.update(self, new, frame_id)
        self.sly_label = new.get_sly_label()

    def re_activate(self, new: Track, frame_id, new_id=False):
        STrack.re_activate(self, new, frame_id, new_id)
        self.sly_label = new.get_sly_label()


class _BoTSORT(BoTSORT):
    def __init__(self, args):
        super().__init__(args)

        # for type hinting
        self.tracked_stracks = []  # type: list[Track]
        self.lost_stracks = []  # type: list[Track]
        self.removed_stracks = []  # type: list[Track]

        # for pylint
        self.frame_id = 0

    def _init_track(
        self, dets: np.ndarray, scores: np.ndarray, labels: List[Label], features: np.ndarray
    ):
        if len(dets) == 0:
            return []
        if self.args.with_reid:
            return [
                Track(Track.tlbr_to_tlwh(tlbr), s, f, l)
                for (tlbr, s, f, l) in zip(dets, scores, features, labels)
            ]
        return [
            Track(Track.tlbr_to_tlwh(tlbr), s, None, l)
            for (tlbr, s, l) in zip(dets, scores, labels)
        ]

    def get_dists(self, tracks: List[Track], detections: List[Track]):
        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(tracks, detections)

        # if not self.args.mot20:
        ious_dists = matching.fuse_score(ious_dists, detections)

        # Remove detections with different shapes
        for track_i, track in enumerate(tracks):
            for det_i, det in enumerate(detections):
                if track.get_sly_label().geometry.name() != det.get_sly_label().geometry.name():
                    ious_dists[track_i, det_i] = 1.0

        if not self.args.with_reid:
            return ious_dists

        ious_dists_mask = ious_dists > self.args.proximity_thresh
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        # raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.args.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        # Popular ReID method (JDE / FairMOT)
        # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
        # emb_dists = dists

        # IoU making ReID
        # dists = matching.embedding_distance(strack_pool, detections)
        # dists[ious_dists_mask] = 1.0
        return dists

    def update(self, detections_: List[Detection], img) -> List[Track]:
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections_):
            scores = np.array([det.confidence for det in detections_])
            bboxes = np.array([det.tlbr()[:4] for det in detections_])
            labels = np.array([det.sly_label for det in detections_])
            features = np.array([det.feature for det in detections_])

            # Remove bad detections
            lowest_inds = scores > self.args.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            labels = labels[lowest_inds]
            features = features[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            labels_keep = labels[remain_inds]
            features_keep = features[remain_inds]
        else:
            bboxes = []
            scores = []
            labels = []
            dets = []
            scores_keep = []
            labels_keep = []
            features_keep = []

        """Extract embeddings """
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        """Detections"""
        detections = self._init_track(dets, scores_keep, labels_keep, features_keep)

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool: List[Track] = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track: Track = strack_pool[itracked]
            det: Track = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(scores):
            inds_second = scores < self.args.track_high_thresh
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            labels_second = labels[inds_second]
            features_second = features[inds_second]
        else:
            dets_second = []
            scores_second = []
            labels_second = []
            features_second = []

        # association the untrack to the low score detections
        detections_second = self._init_track(
            dets_second, scores_second, labels_second, features_second
        )

        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks


class BoTTracker(BaseTracker):
    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        super().__init__(settings=settings)
        self.tracker = _BoTSORT(self.args)

        if self.args.with_reid:
            if not Path(self.args.fast_reid_weights).exists():
                logger.info("Downloading ReID weights...")

                with requests.get(self.args.fast_reid_weights_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.args.fast_reid_weights, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

    def default_settings(self):
        return {
            "name": None,
            "ablation": None,
            # Tracking
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.5,
            "min_box_area": 10,
            "fuse_score": False,
            # CMC
            "cmc_method": "sparseOptFlow",
            "gmc_config": None,
            # ReID
            "with_reid": False,
            "fast_reid_config": f"{Path(__file__).parent}/fast_reid/configs/MOT17/sbs_S50.yml",
            "fast_reid_weights": f"pretrained/yolo7x.pt",
            "fast_reid_weights_url": r"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
        }

    def track(
        self,
        source: Union[List[np.ndarray], List[str], str],
        frame_to_annotation: Dict[int, Annotation],
        frame_shape: Tuple[int, int],
        pbar_cb=None,
    ) -> Annotation:
        """
        Track objects in the video using BoTSort algorithm.

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
            from supervisely.nn.tracker import BoTTracker

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
            tracker = BoTTracker()
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
        logger.info("Starting BoTSort tracking...")
        for frame_index, img in enumerate(self.frames_generator(source)):
            self.update(img, frame_to_annotation[frame_index], frame_index, tracks_data=tracks_data)

            if pbar_cb is not None:
                pbar_cb()

        return self.get_annotation(
            tracks_data=tracks_data,
            frame_shape=frame_shape,
            frames_count=len(frame_to_annotation),
        )

    def update(
        self, img, annotation: Annotation, frame_index, tracks_data: Dict[int, List[Dict]] = None
    ):
        pred, sly_labels = self.convert_annotation(annotation)

        detections = [Detection(p[:4], p[4], None, label) for p, label in zip(pred, sly_labels)]

        self.tracker.update(detections, img)

        if tracks_data is None:
            tracks_data = {}
        self.update_track_data(
            tracks_data=tracks_data,
            tracks=self.tracker.tracked_stracks,
            frame_index=frame_index,
        )

        return tracks_data
