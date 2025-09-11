from typing import List, Union, Dict, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

import supervisely as sly
from supervisely.nn.model.prediction import Prediction
from supervisely.annotation.label import LabelingStatus
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
            frame_figures.append(sly.VideoFigure(video_object, rect, frame_idx, track_id=str(track_id), status=LabelingStatus.AUTO))
        
        frames.append(sly.Frame(frame_idx, frame_figures))

    objects = list(video_objects.values()) 
    return VideoAnnotation(
        img_size=frame_shape,
        frames_count=len(predictions),
        objects=sly.VideoObjectCollection(objects),
        frames=sly.FrameCollection(frames)
    )
    
def video_annotation_to_mot(
    annotation: VideoAnnotation,
    output_path: Union[str, Path] = None,
    class_to_id_mapping: Dict[str, int] = None
) -> Union[str, List[str]]:
    """
    Convert Supervisely VideoAnnotation to MOT format.
    MOT format: frame_id,track_id,left,top,width,height,confidence,class_id,visibility
    """
    mot_lines = []
    
    # Create default class mapping if not provided
    if class_to_id_mapping is None:
        unique_classes = set()
        for frame in annotation.frames:
            for figure in frame.figures:
                unique_classes.add(figure.video_object.obj_class.name)
        class_to_id_mapping = {cls_name: idx + 1 for idx, cls_name in enumerate(sorted(unique_classes))}
    
    # Extract tracks
    for frame in annotation.frames:
        frame_id = frame.index + 1  # MOT uses 1-based frame indexing
        
        for figure in frame.figures:
            # Get track ID from VideoFigure.track_id (official API)
            if figure.track_id is not None:
                track_id = int(figure.track_id)
            else:
                track_id = figure.video_object.key().int
            
            # Get bounding box
            if isinstance(figure.geometry, sly.Rectangle):
                bbox = figure.geometry
            else:
                bbox = figure.geometry.to_bbox()
            
            left = bbox.left
            top = bbox.top
            width = bbox.width
            height = bbox.height
            
            # Get class ID
            class_name = figure.video_object.obj_class.name
            class_id = class_to_id_mapping.get(class_name, 1)
            
            # Get confidence (default)
            confidence = 1.0
            
            # Visibility (assume visible)
            visibility = 1
            
            # Create MOT line
            mot_line = f"{frame_id},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{confidence:.3f},{class_id},{visibility}"
            mot_lines.append(mot_line)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for line in mot_lines:
                f.write(line + '\n')
        
        logger.info(f"Saved MOT format to: {output_path} ({len(mot_lines)} detections)")
        return str(output_path)
    
    return mot_lines

def mot_to_video_annotation(
    mot_file_path: Union[str, Path],
    img_size: Tuple[int, int] = (1080, 1920),
    class_mapping: Dict[int, str] = None,
    default_class_name: str = "person"
) -> VideoAnnotation:
    """
    Convert MOT format tracking data to Supervisely VideoAnnotation.
    MOT format: frame_id,track_id,left,top,width,height,confidence,class_id,visibility
    """
    mot_file_path = Path(mot_file_path)
    
    if not mot_file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_file_path}")
    
    logger.info(f"Loading MOT data from: {mot_file_path}")
    logger.info(f"Image size: {img_size} (height, width)")
    
    # Default class mapping
    if class_mapping is None:
        class_mapping = {1: default_class_name}
    
    # Parse MOT file
    video_objects = {}  # track_id -> VideoObject
    frames_data = defaultdict(list)  # frame_idx -> list of figures
    max_frame_idx = 0
    img_h, img_w = img_size
    
    with open(mot_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split(',')
                if len(parts) < 6:  # Minimum required fields
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                
                # Optional fields
                confidence = float(parts[6]) if len(parts) > 6 and parts[6] != '-1' else 1.0
                class_id = int(parts[7]) if len(parts) > 7 and parts[7] != '-1' else 1
                visibility = float(parts[8]) if len(parts) > 8 and parts[8] != '-1' else 1.0
                
                frame_idx = frame_id - 1  # Convert to 0-based indexing
                max_frame_idx = max(max_frame_idx, frame_idx)
                
                # Skip low confidence detections
                if confidence < 0.1:
                    continue
                
                # Calculate coordinates with safer clipping
                right = left + width
                bottom = top + height
                
                # Clip to image boundaries
                left = max(0, int(left))
                top = max(0, int(top))
                right = min(int(right), img_w - 1)
                bottom = min(int(bottom), img_h - 1)
                
                # Skip invalid boxes
                if right <= left or bottom <= top:
                    continue
                
                # Get class name
                class_name = class_mapping.get(class_id, default_class_name)
                
                # Create VideoObject if not exists
                if track_id not in video_objects:
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    video_objects[track_id] = sly.VideoObject(obj_class)
                
                video_object = video_objects[track_id]
                
                # Create rectangle and figure with track_id
                rect = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                figure = sly.VideoFigure(video_object, rect, frame_idx, track_id=str(track_id))
                
                frames_data[frame_idx].append(figure)
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipped invalid MOT line {line_num}: {line} - {e}")
                continue
    
    # Create frames
    frames = []
    if frames_data:
        frames_count = max(frames_data.keys()) + 1
        
        for frame_idx in range(frames_count):
            figures = frames_data.get(frame_idx, [])
            frames.append(sly.Frame(frame_idx, figures))
    else:
        frames_count = 1
        frames = [sly.Frame(0, [])]
    
    # Create VideoAnnotation
    objects = list(video_objects.values())
    
    annotation = VideoAnnotation(
        img_size=img_size,
        frames_count=frames_count,
        objects=sly.VideoObjectCollection(objects),
        frames=sly.FrameCollection(frames)
    )
    
    logger.info(f"Created VideoAnnotation with {len(objects)} tracks and {frames_count} frames")
    
    return annotation
