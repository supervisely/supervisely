import supervisely as sly
from supervisely.nn.tracker.base_tracker import BaseTracker
from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT
from supervisely import Annotation, VideoAnnotation
from types import SimpleNamespace
from typing import List, Dict, Tuple, Any
import numpy as np
import yaml
import os
from pathlib import Path
from supervisely import logger


class BotSortTracker(BaseTracker):
    
    def __init__(self, settings: dict = None, device: str = None):
        super().__init__()
        
        # Load default settings from YAML file
        self.settings = self._load_default_settings()
        
        # Override with user settings if provided
        if settings:
            self.settings.update(settings)
        
        args = SimpleNamespace(**self.settings)
            
        self.tracker = BoTSORT(args=args)
        self.device = device
        
        # State for accumulating results
        self.frame_tracks = []
        self.obj_classes = {}   # class_id -> ObjClass
        self.current_frame = 0
        self.class_ids = {}  # class_name -> class_id mapping
        self.frame_shape = () 

    def _load_default_settings(self) -> dict:
        """Load default settings from YAML file in the same directory."""
        current_dir = Path(__file__).parent
        config_path = current_dir / "botsort_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def update(self, frame: np.ndarray, annotation: Annotation) -> List[Dict[str, Any]]:
        """Update tracker and return tracks for current frame."""
        self.frame_shape = frame.shape[:2]
        self._update_obj_classes(annotation)
        detections = self._convert_annotation(annotation)
        output_stracks = self.tracker.update(detections, frame)
        tracks = self._stracks_to_tracks(output_stracks)
        
        # Store tracks for VideoAnnotation creation
        if tracks:
            self.frame_tracks.append(tracks)
        
        self.current_frame += 1
        return tracks
    
    def reset(self):
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
    
    def _stracks_to_tracks(self, output_stracks) -> List[Dict[str, Any]]:
        """Convert BoTSORT output tracks to standard dictionary format."""
        tracks = []
        
        for strack in output_stracks:
            # BoTSORT may store class info in different attributes
            # Try to get class_id from various possible sources
            class_id = 0  # default
            
            if hasattr(strack, 'cls') and strack.cls != -1:
                # cls should contain the numeric ID we passed in
                class_id = int(strack.cls)
            elif hasattr(strack, 'class_id'):
                class_id = int(strack.class_id)
            
            track = {
                'track_id': strack.track_id,
                'bbox': strack.tlbr.tolist(),  # [x1, y1, x2, y2]
                'class_id': class_id,
                'score': getattr(strack, 'score', 1.0)
            }
            tracks.append(track)
        
        return tracks
    
    def _update_obj_classes(self, annotation: Annotation):
        """Extract and store object classes from annotation."""
        for label in annotation.labels:
            class_name = label.obj_class.name
            class_id = len(self.class_ids)
            self.obj_classes[class_id] = label.obj_class
            self.class_ids[class_name] = class_id

    def _create_video_annotation(self) -> VideoAnnotation:
        """Convert accumulated tracking results to Supervisely VideoAnnotation."""
        img_h, img_w = self.frame_shape
        video_objects = {}  # track_id -> VideoObject
        frames = []
        
        for frame_idx, tracks in self.frame_tracks.items():
            frame_figures = []
            
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']  # [x1, y1, x2, y2]
                class_id = track['class_id']
                
                # Clip bbox to image boundaries
                x1, y1, x2, y2 = bbox
                dims = np.array([img_w, img_h, img_w, img_h]) - 1
                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, dims)
                
                # Get or create VideoObject
                if track_id not in video_objects:
                    obj_class = self.obj_classes.get(class_id)
                    if obj_class is None:
                        continue  # Skip if class not found
                    video_objects[track_id] = sly.VideoObject(obj_class)
                
                video_object = video_objects[track_id]
                rect = sly.Rectangle(top=y1, left=x1, bottom=y2, right=x2)
                frame_figures.append(sly.VideoFigure(video_object, rect, frame_idx))
            
            frames.append(sly.Frame(frame_idx, frame_figures))

        objects = list(video_objects.values())
        frames_count = max(self.frame_tracks.keys()) + 1 if self.frame_tracks else 0
        
        return VideoAnnotation(
            img_size=self.frame_shape,
            frames_count=frames_count,
            objects=sly.VideoObjectCollection(objects),
            frames=sly.FrameCollection(frames)
        )

