
from typing import List
import numpy as np

import supervisely as sly
from supervisely.nn.model.prediction import Prediction
from supervisely import VideoAnnotation

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