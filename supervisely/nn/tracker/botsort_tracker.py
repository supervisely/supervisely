import supervisely as sly
from supervisely.nn.tracker.base_tracker import BaseTracker
from supervisely import Annotation, VideoAnnotation
from supervisely.annotation.label import LabelingStatus
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import yaml
import os
from pathlib import Path
from supervisely import logger
from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT


@dataclass
class TrackedObject:
    """
    Data class representing a tracked object in a single frame.
    
    Args:
        track_id: Unique identifier for the track
        det_id: Detection ID for mapping back to original annotation
        bbox: Bounding box coordinates in format [x1, y1, x2, y2]
        class_name: String class name
        class_sly_id: Supervisely class ID (from ObjClass.sly_id)
        score: Confidence score of the detection/track
    """
    track_id: int
    det_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    class_sly_id: Optional[int]  # Supervisely class ID
    score: float


class BotSortTracker(BaseTracker):
    
    def __init__(self, settings: dict = None, device: str = None):
        super().__init__(settings=settings, device=device)

        from supervisely.nn.tracker import TRACKING_LIBS_INSTALLED
        if not TRACKING_LIBS_INSTALLED:
            raise ImportError(
                "Tracking dependencies are not installed. "
                "Please install supervisely with `pip install supervisely[tracking]`."
            )
        
        # Load default settings from YAML file
        self.settings = self._load_default_settings()
        
        # Override with user settings if provided
        if settings:
            self.settings.update(settings)
        
        args = SimpleNamespace(**self.settings)
        args.name = "BotSORT"
        args.device = self.device
            
        self.tracker = BoTSORT(args=args)
        
        # State for accumulating results
        self.frame_tracks = []
        self.obj_classes = {}   # class_id -> ObjClass
        self.current_frame = 0
        self.class_ids = {}  # class_name -> class_id mapping
        self.frame_shape = () 

    def _load_default_settings(self) -> dict:
        """Internal method: calls classmethod"""
        return self.get_default_params()

    def update(self, frame: np.ndarray, annotation: Annotation) -> List[Dict[str, Any]]:
        """Update tracker and return list of matches for current frame."""
        self.frame_shape = frame.shape[:2]
        self._update_obj_classes(annotation)
        detections = self._convert_annotation(annotation)
        output_stracks, detection_track_map = self.tracker.update(detections, frame)
        tracks = self._stracks_to_tracks(output_stracks, detection_track_map)
        
        # Store tracks for VideoAnnotation creation
        self.frame_tracks.append(tracks)
        self.current_frame += 1
        
        matches = []
        for pair in detection_track_map:
            det_id = pair["det_id"]
            track_id = pair["track_id"]

            if track_id is not None:
                match = {
                    "track_id": track_id,
                    "label": annotation.labels[det_id]
                }
                matches.append(match)
            
        return matches
    
    def reset(self) -> None:
        super().reset()
        self.frame_tracks = []
        self.obj_classes = {}
        self.current_frame = 0
        self.class_ids = {}
        self.frame_shape = ()

    def track(self, frames: List[np.ndarray], annotations: List[Annotation]) -> VideoAnnotation:
        """Track objects through sequence of frames and return VideoAnnotation."""
        if len(frames) != len(annotations):
            raise ValueError("Number of frames and annotations must match")
        
        self.reset()
        
        # Process each frame
        for frame_idx, (frame, annotation) in enumerate(zip(frames, annotations)):
            self.current_frame = frame_idx
            self.update(frame, annotation)
        
        # Convert accumulated tracks to VideoAnnotation
        return self._create_video_annotation()
        
    def _convert_annotation(self, annotation: Annotation) -> np.ndarray:
        """Convert Supervisely annotation to BoTSORT detection format."""
        detections_list = []

        for label in annotation.labels:
            if label.tags.get("confidence", None) is not None:
                confidence = label.tags.get("confidence").value
            elif label.tags.get("conf", None) is not None:
                confidence = label.tags.get("conf").value
            else:
                confidence = 1.0
                logger.debug(
                    f"Label {label.obj_class.name} does not have confidence tag, using default value 1.0"
                )

            rectangle = label.geometry.to_bbox()
            
            class_name = label.obj_class.name
            class_id = self.class_ids[class_name]
            
            detection = [
                rectangle.left,    # x1
                rectangle.top,     # y1 
                rectangle.right,   # x2
                rectangle.bottom,  # y2   
                confidence,        # score
                class_id,          # class_id as number
            ]
            detections_list.append(detection)

        if detections_list:
            return np.array(detections_list, dtype=np.float32)
        else:
            return np.zeros((0, 6), dtype=np.float32)
    
    def _stracks_to_tracks(self, output_stracks, detection_track_map) -> List[TrackedObject]:
        """Convert BoTSORT output tracks to TrackedObject dataclass instances."""
        tracks = []
        
        id_to_name = {v: k for k, v in self.class_ids.items()}
        
        track_id_to_det_id = {}
        for pair in detection_track_map:
            det_id = pair["det_id"]
            track_id = pair["track_id"]
            track_id_to_det_id[track_id] = det_id 
        
        for strack in output_stracks:
            # BoTSORT may store class info in different attributes
            # Try to get class_id from various possible sources
            class_id = 0  # default
            
            if hasattr(strack, 'cls') and strack.cls != -1:
                # cls should contain the numeric ID we passed in
                class_id = int(strack.cls)
            elif hasattr(strack, 'class_id'):
                class_id = int(strack.class_id)
            
            class_name = id_to_name.get(class_id, "unknown")
            
            # Get Supervisely class ID from stored ObjClass
            class_sly_id = None
            if class_name in self.obj_classes:
                obj_class = self.obj_classes[class_name]
                class_sly_id = obj_class.sly_id
            
            track = TrackedObject(
                track_id=strack.track_id,
                det_id=track_id_to_det_id.get(strack.track_id),
                bbox=strack.tlbr.tolist(),  # [x1, y1, x2, y2]
                class_name=class_name,
                class_sly_id=class_sly_id,
                score=getattr(strack, 'score', 1.0)
            )
            tracks.append(track)
        
        return tracks
        
    def _update_obj_classes(self, annotation: Annotation):
        """Extract and store object classes from annotation."""
        for label in annotation.labels:
            class_name = label.obj_class.name
            if class_name not in self.obj_classes:
                self.obj_classes[class_name] = label.obj_class
        
            if class_name not in self.class_ids:
                self.class_ids[class_name] = len(self.class_ids)


    def _create_video_annotation(self) -> VideoAnnotation:
        """Convert accumulated tracking results to Supervisely VideoAnnotation."""
        img_h, img_w = self.frame_shape
        video_objects = {}  # track_id -> VideoObject
        frames = []
        
        for frame_idx, tracks in enumerate(self.frame_tracks):
            frame_figures = []
            
            for track in tracks:
                track_id = track.track_id
                bbox = track.bbox  # [x1, y1, x2, y2]
                class_name = track.class_name
                
                # Clip bbox to image boundaries
                x1, y1, x2, y2 = bbox
                dims = np.array([img_w, img_h, img_w, img_h]) - 1
                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, dims)
                
                # Get or create VideoObject
                if track_id not in video_objects:
                    obj_class = self.obj_classes.get(class_name)
                    if obj_class is None:
                        continue  # Skip if class not found
                    video_objects[track_id] = sly.VideoObject(obj_class)
                
                video_object = video_objects[track_id]
                rect = sly.Rectangle(top=y1, left=x1, bottom=y2, right=x2)
                frame_figures.append(sly.VideoFigure(video_object, rect, frame_idx, track_id=str(track_id), status=LabelingStatus.AUTO))
            
            frames.append(sly.Frame(frame_idx, frame_figures))

        objects = list(video_objects.values())

        
        return VideoAnnotation(
            img_size=self.frame_shape,
            frames_count=len(self.frame_tracks),
            objects=sly.VideoObjectCollection(objects),
            frames=sly.FrameCollection(frames)
        )
        
    @property    
    def video_annotation(self) -> VideoAnnotation:
        """Return the accumulated VideoAnnotation."""
        if not self.frame_tracks:
            error_msg = (
                "No tracking data available. "
                "Please run tracking first using track() method or process frames with update()."
            )
            raise ValueError(error_msg)
                
        return self._create_video_annotation()

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Public API: get default params WITHOUT creating instance."""
        current_dir = Path(__file__).parent
        config_path = current_dir / "botsort/botsort_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

