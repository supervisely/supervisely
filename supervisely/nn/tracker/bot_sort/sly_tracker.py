from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import requests

from supervisely import Annotation, Label
from supervisely.nn.tracker.ann_keeper import AnnotationKeeper
from supervisely.nn.tracker.tracker import BaseDetection, BaseTrack, BaseTracker
from supervisely.sly_logger import logger

from .tracker import matching
from .tracker.mc_bot_sort import (
    BoTSORT,
    STrack,
    TrackState,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)


class Detection(BaseDetection):
    def __init__(self, sly_label: Label, tlwh, confidence, feature=None):
        self.tlwh = tlwh
        self.tlbr = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
        self.confidence = confidence
        self.feature = feature
        self.sly_label = sly_label

    def get_sly_label(self):
        return self.sly_label


class Track(BaseTrack, STrack):
    def __init__(self, sly_label: Label, tlwh, confidence, feature=None):
        STrack.__init__(self, tlwh, confidence, sly_label.obj_class.name, feature)
        self.sly_label = sly_label

    def get_sly_label(self):
        return self.sly_label

    def clean_sly_label(self):
        self.sly_label = None

    def update(self, detection, frame_id):
        super().update(detection, frame_id)
        self.sly_label = detection.get_sly_label()


class _BoTSORT(BoTSORT):
    def __init__(self, args):
        super().__init__(args)
        self.tracked_stracks = []  # type: list[Track]
        self.lost_stracks = []  # type: list[Track]
        self.removed_stracks = []  # type: list[Track]

    def update(self, detections_: List[Detection], img) -> List[Track]:
        output_results = np.array(
            [
                [det.tlbr[0], det.tlbr[1], det.tlbr[2], det.tlbr[3], det.confidence]
                for det in detections_
            ]
        )

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
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

        """Extract embeddings """
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            """Detections"""
            if self.args.with_reid:
                detections = [
                    Track(l, STrack.tlbr_to_tlwh(tlbr), s, f)
                    for (tlbr, s, l, f) in zip(dets, scores_keep, labels_keep, features_keep)
                ]
            else:
                detections = [
                    Track(l, STrack.tlbr_to_tlwh(tlbr), s)
                    for (tlbr, s, l) in zip(dets, scores_keep, labels_keep)
                ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        # if not self.args.mot20:
        ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            labels_second = labels[inds_second]
        else:
            dets_second = []
            scores_second = []
            labels_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                Track(l, STrack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s, l) in zip(dets_second, scores_second, labels_second)
            ]
        else:
            detections_second = []

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
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
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
            # ReID
            "with_reid": False,
            "fast_reid_config": r"supervisely/nn/tracker/bot_sort/fast_reid/configs/MOT17/sbs_S50.yml",
            "fast_reid_weights": r"supervisely/nn/tracker/bot_sort/pretrained/yolo7x.pt",
            "fast_reid_weights_url": r"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
        }

    def track(
        self,
        images: Union[List[np.ndarray], List[str]],
        frame_to_annotation: Dict[int, Annotation],
        annotation_keeper: AnnotationKeeper = None,
        pbar_cb=None,
    ) -> Dict[int, Dict]:
        if len(images) != len(frame_to_annotation):
            raise ValueError("Number of images and annotations should be the same")

        tracks_data = {}
        logger.info("Starting BoTSort tracking...")
        for image_index, frame_index in enumerate(frame_to_annotation.keys()):
            detections = []
            img = images[image_index]
            if isinstance(img, str):
                img = cv2.imread(img)

            pred, sly_labels = self.convert_annotation(frame_to_annotation[frame_index])

            detections = [Detection(label, p[:4], p[4]) for p, label in zip(pred, sly_labels)]

            self.tracker.update(detections, img)

            self.update_track_data(
                tracks_data=tracks_data,
                tracks=self.tracker.tracked_stracks,
                frame_index=frame_index,
                img_size=img.shape[:2],
            )

            if pbar_cb is not None:
                pbar_cb()

        logger.debug(
            "Tracks",
            extra={
                "tracks": [
                    {"track_id": track.track_id, "cls_hist": track.cls_hist}
                    for track in self.tracker.tracked_stracks
                ]
            },
        )

        tracks_data = self.clear_empty_ids(tracker_annotations=tracks_data)

        if annotation_keeper is not None:
            annotation_keeper.add_figures_by_frames(tracks_data)
        return tracks_data
