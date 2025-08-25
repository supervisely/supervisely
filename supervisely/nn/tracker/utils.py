
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
    default_class_name: str = "person",
    debug: bool = False
) -> VideoAnnotation:
    """Convert MOT format tracking data to Supervisely VideoAnnotation."""
    mot_file_path = Path(mot_file_path)
    
    if not mot_file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_file_path}")
    
    logger.info(f"Loading MOT data from: {mot_file_path}")
    logger.info(f"Image size: {img_size} (height, width)")
    
    if img_size[0] < img_size[1]:
        logger.warning(f"Suspicious image size: height ({img_size[0]}) < width ({img_size[1]}). "
                      "Make sure you pass (height, width), not (width, height)!")
    
    # Parse MOT file
    tracks_by_frame = defaultdict(list)
    video_objects = {}
    max_frame_id = 0
    
    # Create TagMeta for storing track_id
    track_id_meta = sly.TagMeta('track_id', sly.TagValueType.ANY_NUMBER)
    img_h, img_w = img_size
    
    with open(mot_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                frame_id = int(parts[0].strip())
                track_id = int(parts[1].strip())
                x = float(parts[2].strip())
                y = float(parts[3].strip())
                width = float(parts[4].strip())
                height = float(parts[5].strip())
                confidence = float(parts[6].strip())
                class_id = int(parts[7].strip()) if parts[7].strip() != '-1' else 1
                
                frame_idx = frame_id - 1
                max_frame_id = max(max_frame_id, frame_idx)
                
                if confidence < 0.1:
                    continue
                
                # IMPROVED CLIPPING: More aggressive boundary enforcement
                left = max(0, min(x, img_w - 2))
                top = max(0, min(y, img_h - 2))
                right = max(left + 1, min(x + width, img_w - 1))
                bottom = max(top + 1, min(y + height, img_h - 1))
                
                # Convert to integers
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                
                # Skip invalid boxes
                if right <= left or bottom <= top:
                    continue
                
                # Determine class name
                if class_mapping and class_id in class_mapping:
                    class_name = class_mapping[class_id]
                else:
                    class_name = default_class_name
                
                # Create VideoObject with track_id in tags
                if track_id not in video_objects:
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    track_tag = sly.VideoTag(track_id_meta, value=track_id)
                    video_objects[track_id] = sly.VideoObject(
                        obj_class, 
                        tags=sly.VideoTagCollection([track_tag])
                    )
                
                tracks_by_frame[frame_idx].append({
                    'track_id': track_id,
                    'bbox': (left, top, right, bottom),
                    'video_object': video_objects[track_id],
                    'confidence': confidence,
                })
                
            except (ValueError, IndexError) as e:
                if debug:
                    logger.warning(f"Line {line_num}: Failed to parse: {e}")
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
                
                try:
                    rect = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                    figure = sly.VideoFigure(video_object, rect, frame_idx)
                    frame_figures.append(figure)
                except Exception as e:
                    if debug:
                        logger.warning(f"Failed to create figure on frame {frame_idx}: {e}")
                    continue
        
        frames.append(sly.Frame(frame_idx, frame_figures))
    
    # CREATE VIDEOANNOTATION WITHOUT VALIDATION
    objects_collection = sly.VideoObjectCollection(list(video_objects.values()))
    frames_collection = sly.FrameCollection(frames)
    
    # Manual creation to skip bounds validation
    video_annotation = VideoAnnotation.__new__(VideoAnnotation)
    video_annotation._img_size = img_size
    video_annotation._frames_count = max_frame_id + 1
    video_annotation._objects = objects_collection
    video_annotation._frames = frames_collection
    video_annotation._tags = sly.VideoTagCollection()
    video_annotation._description = ""
    video_annotation._key = sly.generate_free_name([], "", with_ext=False)
    
    logger.info(f"Created VideoAnnotation with {len(video_objects)} objects and {len(frames)} frames")
    
    return video_annotation