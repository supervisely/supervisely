from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import requests

from supervisely import Annotation, Label
from supervisely.nn.tracker.tracker import BaseDetection as Detection
from supervisely.nn.tracker.tracker import BaseTrack, BaseTracker
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


class Track(BaseTrack, STrack):
    """
    Enhanced Track class that combines BaseTrack and STrack functionality
    with Supervisely label support.
    """
    
    def __init__(self, tlwh, confidence: float, cls_name: str, feature=None, sly_label: Label = None):
        """
        Initialize track with bounding box, confidence, class and optional feature.
        
        Args:
            tlwh: Bounding box in tlwh format (top-left-x, top-left-y, width, height)
            confidence: Detection confidence score
            cls_name: Class name string
            feature: Optional ReID feature vector
            sly_label: Supervisely label object
        """
        self.track_id = None
        STrack.__init__(self, tlwh, confidence, cls_name, feature)
        self.sly_label = sly_label

    def get_sly_label(self) -> Label:
        """Get associated Supervisely label."""
        return self.sly_label

    def update(self, new: Track, frame_id: int):
        """
        Update track with new detection.
        
        Args:
            new: New track detection
            frame_id: Current frame ID
        """
        STrack.update(self, new, frame_id)
        self.sly_label = new.get_sly_label()

    def re_activate(self, new: Track, frame_id: int, new_id: bool = False):
        """
        Re-activate lost track with new detection.
        
        Args:
            new: New track detection
            frame_id: Current frame ID
            new_id: Whether to assign new track ID
        """
        STrack.re_activate(self, new, frame_id, new_id)
        self.sly_label = new.get_sly_label()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
        
class _BoTSORT(BoTSORT):
    """
    Enhanced BoTSORT tracker with Supervisely integration.
    Mimics the exact behavior of direct mc_bot_sort.py usage.
    """
    
    def __init__(self, args):
        super().__init__(args)

        # Type hints for tracked objects
        self.tracked_stracks = []  # type: list[Track]
        self.lost_stracks = []  # type: list[Track]
        self.removed_stracks = []  # type: list[Track]

        # Frame counter
        self.frame_id = 0

    def update(self, output_results: np.ndarray, img: np.ndarray, sly_labels: List[Label] = None) -> List[Track]:
        """
        Update tracker with new detections in the exact same format as mc_bot_sort.py.
        
        Args:
            output_results: numpy array (N,6) [x1,y1,x2,y2,score,class] - EXACT format as mc_bot_sort.py
            img: Current frame image
            sly_labels: Optional list of Supervisely labels for track association
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            # Extract data exactly as in mc_bot_sort.py
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            
            # Map classes to labels if provided
            if sly_labels and len(sly_labels) == len(output_results):
                mapped_labels = sly_labels
                class_names = [label.obj_class.name for label in sly_labels]
            else:
                # Fallback to class IDs as strings
                mapped_labels = [None] * len(output_results)
                class_names = [f"class_{int(cls)}" for cls in classes]

            # Remove bad detections - EXACT logic from mc_bot_sort.py
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            mapped_labels = [mapped_labels[i] for i in range(len(mapped_labels)) if lowest_inds[i]]
            class_names = [class_names[i] for i in range(len(class_names)) if lowest_inds[i]]

            # Find high threshold detections - EXACT logic
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            labels_keep = [mapped_labels[i] for i in range(len(mapped_labels)) if remain_inds[i]]
            class_names_keep = [class_names[i] for i in range(len(class_names)) if remain_inds[i]]
        else:
            bboxes = np.empty((0, 4))
            scores = np.empty(0)
            classes = np.empty(0)
            dets = np.empty((0, 4))
            scores_keep = np.empty(0)
            classes_keep = np.empty(0)
            labels_keep = []
            class_names_keep = []

        # Extract embeddings - EXACT logic
        if self.args.with_reid and len(dets) > 0:
            features_keep = self.encoder.inference(img, dets)
        else:
            features_keep = None

        # Initialize detections - convert to Track objects
        if len(dets) > 0:
            if self.args.with_reid and features_keep is not None:
                detections = [
                    Track(Track.tlbr_to_tlwh(tlbr), s, c, f, l)
                    for (tlbr, s, c, f, l) in zip(dets, scores_keep, class_names_keep, features_keep, labels_keep)
                ]
            else:
                detections = [
                    Track(Track.tlbr_to_tlwh(tlbr), s, c, None, l)
                    for (tlbr, s, c, l) in zip(dets, scores_keep, class_names_keep, labels_keep)
                ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks - EXACT logic
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association, with high score detection boxes - EXACT logic
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes - EXACT logic
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not getattr(self.args, 'mot20', False):
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association, with low score detection boxes - EXACT logic
        if len(scores) > 0:
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
            labels_second = [mapped_labels[i] for i in range(len(mapped_labels)) if inds_second[i]]
            class_names_second = [class_names[i] for i in range(len(class_names)) if inds_second[i]]
        else:
            dets_second = np.empty((0, 4))
            scores_second = np.empty(0)
            classes_second = np.empty(0)
            labels_second = []
            class_names_second = []

        # Association the untrack to the low score detections
        if len(dets_second) > 0:
            detections_second = [
                Track(Track.tlbr_to_tlwh(tlbr), s, c, None, l)
                for (tlbr, s, c, l) in zip(dets_second, scores_second, class_names_second, labels_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
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
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame - EXACT logic
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        
        if not getattr(self.args, 'mot20', False):
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new stracks - EXACT logic
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Step 5: Update state - EXACT logic
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Merge - EXACT logic
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

        # Return only activated tracks - EXACT logic
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


class BoTTracker(BaseTracker):
    """
    BoTSORT tracker wrapper for Supervisely framework.
    Provides multi-object tracking with optional ReID features.
    """
    
    def __init__(self, settings=None):
        """
        Initialize BoTSORT tracker with given settings.
        
        Args:
            settings: Dictionary with tracker configuration parameters
        """
        if settings is None:
            settings = {}
        super().__init__(settings=settings)
        self.tracker = _BoTSORT(self.args)

        # Download ReID weights if needed
        if self.args.with_reid and self.args.reid_model == "fast_reid":
            if not Path(self.args.reid_weights).exists():
                logger.info("Downloading fast-ReID weights...")
                with requests.get(self.args.fast_reid_weights_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.args.reid_weights, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

    def default_settings(self) -> Dict:
        """
        Get default tracker settings.
        
        Returns:
            Dictionary with default configuration parameters
        """
        return {
            "name": None,
            "ablation": None,
            # Tracking thresholds
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.5,
            "min_box_area": 10,
            "fuse_score": False,
            # Camera motion compensation
            "cmc_method": "sparseOptFlow",
            "gmc_config": None,
            # ReID settings
            "with_reid": False,
            "reid_model": "osnet_reid",
            "fast_reid_config": f"{Path(__file__).parent}/fast_reid/configs/MOT17/sbs_S50.yml",
            "reid_weights": None,
            "fast_reid_weights_url": r"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            # Additional settings
            "mot20": False,
            "fp16": False,
            "device": "cpu",
        }

    def convert_annotation(self, annotation_for_frame: Annotation):
        """
        Convert Supervisely annotation to detection format.
        Returns data in EXACT same format as direct mc_bot_sort.py usage: (N,6) [x1,y1,x2,y2,score,class]
        
        Args:
            annotation_for_frame: Supervisely annotation for single frame
            
        Returns:
            Tuple of (detections_array, sly_labels)
        """
        detections_list = []
        sly_labels = []

        for label in annotation_for_frame.labels:
            confidence = 1.0
            if label.tags.get("confidence", None) is not None:
                confidence = label.tags.get("confidence").value
            elif label.tags.get("conf", None) is not None:
                confidence = label.tags.get("conf").value

            rectangle = label.geometry.to_bbox()
            
            # Create detection in EXACT format as mc_bot_sort.py expects: [x1,y1,x2,y2,score,class]
            # Convert class name to numeric ID (simple hash for consistency)
            class_id = hash(label.obj_class.name) % 1000  # Simple class ID generation
            
            detection = [
                rectangle.left,    # x1
                rectangle.top,     # y1 
                rectangle.right,   # x2
                rectangle.bottom,  # y2
                confidence,        # score
                class_id,         # class
            ]

            detections_list.append(detection)
            sly_labels.append(label)

        # Convert to numpy array in EXACT format as mc_bot_sort.py
        if detections_list:
            detections_array = np.array(detections_list, dtype=np.float32)
        else:
            detections_array = np.zeros((0, 6), dtype=np.float32)

        return detections_array, sly_labels

    def track(
        self,
        source: Union[List[np.ndarray], List[str], str],
        frame_to_annotation: Dict[int, Annotation],
        frame_shape: Tuple[int, int],
        pbar_cb=None,
    ) -> Annotation:
        """
        Track objects in video using BoTSORT algorithm.

        Args:
            source: List of images, paths to images or path to video file
            frame_to_annotation: Dictionary mapping frame index to annotations
            frame_shape: Frame dimensions (height, width)
            pbar_cb: Optional progress callback function

        Returns:
            Video annotation with tracked objects

        Raises:
            ValueError: If number of images and annotations don't match

        Usage example:
            ```python
            import supervisely as sly
            from supervisely.nn.tracker import BoTTracker

            api = sly.Api()
            project_id = 12345
            video_id = 12345678
            video_path = "video.mp4"

            # Download video and get info
            video_info = api.video.get_info_by_id(video_id)
            frame_shape = (video_info.frame_height, video_info.frame_width)
            api.video.download_path(id=video_id, path=video_path)

            # Get detections from inference
            task_id = 12345
            session = sly.nn.inference.Session(api, task_id)
            annotations = session.inference_video_id(video_id, 0, video_info.frames_count)
            frame_to_annotation = {i: ann for i, ann in enumerate(annotations)}

            # Run tracking
            tracker = BoTTracker()
            video_ann = tracker.track(video_path, frame_to_annotation, frame_shape)

            # Upload results
            model_meta = session.get_model_meta()
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            project_meta = project_meta.merge(model_meta)
            api.video.annotation.append(video_id, video_ann)
            ```
        """
        if not isinstance(source, str):
            if len(source) != len(frame_to_annotation):
                raise ValueError("Number of images and annotations should be the same")

        tracks_data = {}
        logger.info("Starting BoTSORT tracking...")
        
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
        self, 
        img: np.ndarray, 
        annotation: Annotation, 
        frame_index: int, 
        tracks_data: Dict[int, List[Dict]] = None
    ) -> Dict[int, List[Dict]]:
        """
        Update tracker with single frame data.
        Uses EXACT same format as direct mc_bot_sort.py usage.
        
        Args:
            img: Current frame image
            annotation: Frame annotations
            frame_index: Current frame index
            tracks_data: Dictionary to store tracking results
            
        Returns:
            Updated tracks_data dictionary
        """
        # Convert Supervisely annotation to EXACT mc_bot_sort.py format
        detections_array, sly_labels = self.convert_annotation(annotation)

        # Update tracker using EXACT same interface as mc_bot_sort.py
        tracks = self.tracker.update(detections_array, img, sly_labels)

        # Store tracking results
        if tracks_data is None:
            tracks_data = {}
            
        self.update_track_data(
            tracks_data=tracks_data,
            tracks=tracks,
            frame_index=frame_index,
        )

        return tracks_data