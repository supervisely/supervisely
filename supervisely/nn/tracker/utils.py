
from typing import List, Union, Dict, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

import supervisely as sly
from supervisely.nn.model.prediction import Prediction
from supervisely import VideoAnnotation
from supervisely import logger


def predictions_to_video_annotation(
    predictions: List[Prediction],
) -> VideoAnnotation:
    """
    Convert list of Prediction objects to VideoAnnotation.
    
    Args:
        predictions: List of Prediction objects, one per frame
        
    Returns:
        VideoAnnotation object with tracked objects
        
    Note:
        Each Prediction should have track_ids in annotation.custom_data
        or the track_ids property should be set
    """
    
    if not predictions:
        raise ValueError("Empty predictions list provided")
    
    frame_shape = predictions[0].annotation.img_size
    img_h, img_w = frame_shape
    video_objects = {}  
    frames = []
    
    for pred in predictions:
        frame_figures = []
        frame_idx = pred.frame_index
        
        # Get data using public properties
        boxes = pred.boxes          # Public property - np.array (N, 4) in tlbr format
        classes = pred.classes      # Public property - list of class names
        track_ids = pred.track_ids  # Public property - can be None
        
        # Skip frame if no detections
        if len(boxes) == 0:
            frames.append(sly.Frame(frame_idx, []))
            continue
            
        # Handle case when track_ids is None - generate sequential IDs
        if track_ids is None:
            track_ids = list(range(len(boxes)))
        
        for bbox, class_name, track_id in zip(boxes, classes, track_ids):
            # Clip bbox to image boundaries
            # Note: pred.boxes returns tlbr format (top, left, bottom, right)
            top, left, bottom, right = bbox
            dims = np.array([img_h, img_w, img_h, img_w]) - 1
            top, left, bottom, right = np.clip([top, left, bottom, right], 0, dims)
            
            # Convert to integer coordinates
            top, left, bottom, right = int(top), int(left), int(bottom), int(right)
            
            # Get or create VideoObject
            if track_id not in video_objects:
                # Find obj_class from prediction annotation
                obj_class = None
                for label in pred.annotation.labels:
                    if label.obj_class.name == class_name:
                        obj_class = label.obj_class
                        break
                
                if obj_class is None:
                    # Create obj_class if not found (fallback)
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    
                video_objects[track_id] = sly.VideoObject(obj_class)
            
            video_object = video_objects[track_id]
            rect = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            frame_figures.append(sly.VideoFigure(video_object, rect, frame_idx))
        
        frames.append(sly.Frame(frame_idx, frame_figures))

    objects = list(video_objects.values())
    
    return VideoAnnotation(
        img_size=frame_shape,
        frames_count=len(predictions),
        objects=sly.VideoObjectCollection(objects),
        frames=sly.FrameCollection(frames)
    )
    
    
def mot_to_video_annotation(
    mot_file_path: Union[str, Path],
    img_size: Tuple[int, int] = (1080, 1920),
    class_mapping: Dict[int, str] = None,
    default_class_name: str = "person"
) -> VideoAnnotation:
    """
    Convert MOT format tracking data to Supervisely VideoAnnotation.
    
    Args:
        mot_file_path: Path to MOT format file (.txt)
        img_size: Video frame size as (height, width)
        class_mapping: Optional mapping from class_id to class_name {1: "person", 2: "car"}
        default_class_name: Default class name if no mapping provided
        
    Returns:
        VideoAnnotation object with tracking data
        
    MOT Format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility, [optional]
        
    Example:
        >>> mot_ann = mot_to_video_annotation(
        ...     "gt.txt", 
        ...     img_size=(1080, 1920),
        ...     class_mapping={1: "person", 2: "vehicle"}
        ... )
    """
    mot_file_path = Path(mot_file_path)
    
    if not mot_file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_file_path}")
    
    logger.info(f"Loading MOT data from: {mot_file_path}")
    
    # Parse MOT file
    tracks_by_frame = defaultdict(list)
    video_objects = {}
    max_frame_id = 0

    with open(mot_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                # Parse MOT line
                parts = line.split(',')
                if len(parts) < 9:
                    logger.warning(f"Line {line_num}: Invalid format, expected at least 9 fields, got {len(parts)}")
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                confidence = float(parts[6])
                class_id = int(parts[7]) if parts[7] != '-1' else 1  # Default to class 1 if -1
                visibility = float(parts[8])
                
                # Convert to 0-based frame indexing (MOT uses 1-based)
                frame_idx = frame_id - 1
                max_frame_id = max(max_frame_id, frame_idx)
                
                # Skip if confidence is too low (for predictions)
                if confidence < 0.1:
                    continue
                
                # Convert MOT bbox format (x, y, width, height) to Supervisely (left, top, right, bottom)
                left = int(x)
                top = int(y)
                right = int(x + width)
                bottom = int(y + height)
                
                # Clip to image boundaries
                img_h, img_w = img_size
                left = max(0, min(left, img_w - 1))
                top = max(0, min(top, img_h - 1))
                right = max(left + 1, min(right, img_w))
                bottom = max(top + 1, min(bottom, img_h))
                
                # Determine class name
                if class_mapping and class_id in class_mapping:
                    class_name = class_mapping[class_id]
                else:
                    class_name = default_class_name
                
                # Create or get video object
                if track_id not in video_objects:
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    video_objects[track_id] = sly.VideoObject(obj_class)
                
                # Store track data
                tracks_by_frame[frame_idx].append({
                    'track_id': track_id,
                    'bbox': (left, top, right, bottom),
                    'video_object': video_objects[track_id],
                    'confidence': confidence,
                    'visibility': visibility
                })
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Line {line_num}: Failed to parse MOT data: {e}")
                continue
    
    logger.info(f"Parsed {len(video_objects)} tracks from {max_frame_id + 1} frames")
    
    # Create frames
    frames = []
    for frame_idx in range(max_frame_id + 1):
        frame_figures = []
        
        if frame_idx in tracks_by_frame:
            for track_data in tracks_by_frame[frame_idx]:
                left, top, right, bottom = track_data['bbox']
                video_object = track_data['video_object']
                
                # Create rectangle geometry
                rect = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                
                # Create video figure
                figure = sly.VideoFigure(video_object, rect, frame_idx)
                
                # Add confidence tag if available
                if track_data['confidence'] < 1.0:
                    conf_tag = sly.VideoTag(
                        sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER),
                        value=track_data['confidence']
                    )
                    figure = figure.clone(tags=sly.VideoTagCollection([conf_tag]))
                
                frame_figures.append(figure)
        
        frames.append(sly.Frame(frame_idx, frame_figures))
    
    # Create video annotation
    objects_collection = sly.VideoObjectCollection(list(video_objects.values()))
    frames_collection = sly.FrameCollection(frames)
    
    video_annotation = VideoAnnotation(
        img_size=img_size,
        frames_count=max_frame_id + 1,
        objects=objects_collection,
        frames=frames_collection
    )
    
    logger.info(f"Created VideoAnnotation with {len(video_objects)} objects and {len(frames)} frames")
    
    return video_annotation