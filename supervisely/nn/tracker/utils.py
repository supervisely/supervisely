
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
    
    
def video_annotation_to_mot(
    annotation: VideoAnnotation,
    output_path: Union[str, Path] = None,
    class_to_id_mapping: Dict[str, int] = None
) -> Union[str, List[str]]:
    """
    Convert Supervisely VideoAnnotation to MOT format.
    
    MOT format: frame_id,track_id,left,top,width,height,confidence,class_id,visibility
    
    Args:
        annotation: Supervisely VideoAnnotation object
        output_path: Path to save MOT file (if None, returns lines as list)
        class_to_id_mapping: Mapping from class names to class IDs
        
    Returns:
        If output_path provided: path to saved file
        If output_path is None: list of MOT format strings
        
    Usage:
        # Convert to MOT lines
        mot_lines = video_annotation_to_mot(annotation)
        
        # Save to file
        mot_file_path = video_annotation_to_mot(annotation, "tracks.txt")
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
            # Get track ID from VideoObject key
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
            
            # Get confidence from tags
            confidence = 1.0
            for tag in figure.tags:
                if tag.meta.name.lower() in ['confidence', 'conf', 'score']:
                    confidence = float(tag.value)
                    break
            
            # Visibility (assume fully visible)
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


def video_annotation_to_mot(
    annotation: VideoAnnotation,
    output_path: Union[str, Path] = None,
    class_to_id_mapping: Dict[str, int] = None
) -> Union[str, List[str]]:
    """
    Convert Supervisely VideoAnnotation to MOT format.
    
    MOT format: frame_id,track_id,left,top,width,height,confidence,class_id,visibility
    
    Args:
        annotation: Supervisely VideoAnnotation object
        output_path: Path to save MOT file (if None, returns lines as list)
        class_to_id_mapping: Mapping from class names to class IDs
        
    Returns:
        If output_path provided: path to saved file
        If output_path is None: list of MOT format strings
        
    Usage:
        # Convert to MOT lines
        mot_lines = video_annotation_to_mot(annotation)
        
        # Save to file
        mot_file_path = video_annotation_to_mot(annotation, "tracks.txt")
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
            # Get track ID from VideoObject key
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
            
            # Get confidence from tags
            confidence = 1.0
            for tag in figure.tags:
                if tag.meta.name.lower() in ['confidence', 'conf', 'score']:
                    confidence = float(tag.value)
                    break
            
            # Visibility (assume fully visible)
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


def video_annotation_to_mot(
    annotation: VideoAnnotation,
    output_path: Union[str, Path] = None,
    class_to_id_mapping: Dict[str, int] = None
) -> Union[str, List[str]]:
    """
    Convert Supervisely VideoAnnotation to MOT format.
    
    MOT format: frame_id,track_id,left,top,width,height,confidence,class_id,visibility
    
    Args:
        annotation: Supervisely VideoAnnotation object
        output_path: Path to save MOT file (if None, returns lines as list)
        class_to_id_mapping: Mapping from class names to class IDs
        
    Returns:
        If output_path provided: path to saved file
        If output_path is None: list of MOT format strings
        
    Usage:
        # Convert to MOT lines
        mot_lines = video_annotation_to_mot(annotation)
        
        # Save to file
        mot_file_path = video_annotation_to_mot(annotation, "tracks.txt")
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
            # Get track ID - try saved track_id first, fallback to VideoObject key
            if hasattr(figure.video_object, '_track_id'):
                track_id = figure.video_object._track_id
            else:
                track_id = figure.video_object.key().int
            
            # Use stored original MOT line if available for exact reconstruction
            if hasattr(figure, '_original_mot_line'):
                # Parse and adjust frame_id in stored line
                parts = figure._original_mot_line.split(',')
                parts[0] = str(frame_id)  # Update frame_id to current
                mot_line = ','.join(parts)
                mot_lines.append(mot_line)
                continue
            
            # Fallback: reconstruct from geometry
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
    
    Args:
        mot_file_path: Path to MOT format file
        img_size: Image size as (height, width)
        class_mapping: Mapping from class IDs to class names
        default_class_name: Default class name if mapping not provided
        
    Returns:
        Supervisely VideoAnnotation object
        
    Usage:
        # Load MOT file
        annotation = mot_to_video_annotation("gt.txt", img_size=(1080, 1920))
        
        # With class mapping
        class_map = {1: "person", 2: "car", 3: "bike"}
        annotation = mot_to_video_annotation("gt.txt", class_mapping=class_map)
    """
    mot_file_path = Path(mot_file_path)
    
    if not mot_file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_file_path}")
    
    logger.info(f"Loading MOT data from: {mot_file_path}")
    logger.info(f"Image size: {img_size} (height, width)")
    
    # Validate image size
    if img_size[0] < img_size[1]:
        logger.warning(f"Suspicious image size: height ({img_size[0]}) < width ({img_size[1]}). "
                      "Make sure you pass (height, width), not (width, height)!")
    
    # Default class mapping
    if class_mapping is None:
        class_mapping = {1: default_class_name}
    
    # Create frames and video objects exactly like in user's tracker
    video_objects = {}  # track_id -> VideoObject (same as user's tracker)
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
                
                # Calculate and clip coordinates to be safe
                right = min(left + width, img_w - 1)
                bottom = min(top + height, img_h - 1)
                left = max(0, int(left))
                top = max(0, int(top))
                right = int(right)
                bottom = int(bottom)
                
                # Skip obviously invalid boxes
                if right <= left or bottom <= top:
                    continue
                
                # Get class name
                class_name = class_mapping.get(class_id, default_class_name)
                
                # Create VideoObject with preserved track_id and MOT data  
                if track_id not in video_objects:
                    obj_class = sly.ObjClass(class_name, sly.Rectangle)
                    video_object = sly.VideoObject(obj_class)
                    video_object._track_id = track_id
                    video_object._mot_data = {}  # Store MOT data by frame
                    video_objects[track_id] = video_object
                
                video_object = video_objects[track_id]
                
                # Store original MOT data for this frame
                video_object._mot_data[frame_idx] = {
                    'left': left, 'top': top, 'width': width, 'height': height,
                    'confidence': confidence, 'class_id': class_id, 'visibility': visibility
                }
                
                video_object = video_objects[track_id]
                
                # Create rectangle geometry
                rect = sly.Rectangle(top=int(top), left=int(left), bottom=int(bottom), right=int(right))
                
                # Create figure (VideoFigure doesn't accept tags in constructor)
                figure = sly.VideoFigure(video_object, rect, frame_idx)
                frames_data[frame_idx].append(figure)
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipped invalid MOT line {line_num}: {line} - {e}")
                continue
    
    # Create frames (only for frames that have data)
    frames = []
    if frames_data:
        # Use actual max frame from data
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
