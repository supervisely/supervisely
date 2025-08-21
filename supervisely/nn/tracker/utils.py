from typing import List
import numpy as np

import supervisely as sly
from supervisely.nn.model.prediction import Prediction
from supervisely import VideoAnnotation

def predictions_to_video_annotation(
    predictions: List[Prediction],
) -> VideoAnnotation:
    
    frame_shape = predictions[0].annotation.img_size
    img_h, img_w = frame_shape
    video_objects = {}  
    frames = []
    
    for pred in predictions:
        frame_figures = []
        
        for frame_idx, track_id, bbox, class_name in zip(
            pred.frame_index,
            pred._track_ids,
            pred._boxes,  
            pred._classes
        ):
            # Clip bbox to image boundaries
            x1, y1, x2, y2 = bbox
            dims = np.array([img_w, img_h, img_w, img_h]) - 1
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, dims)
            
            # Get or create VideoObject
            if track_id not in video_objects:
                obj_class = pred.obj_classes.get(class_name)
                if obj_class is None:
                    continue
                video_objects[track_id] = sly.VideoObject(obj_class)
            video_object = video_objects[track_id]
            rect = sly.Rectangle(top=y1, left=x1, bottom=y2, right=x2)
            frame_figures.append(sly.VideoFigure(video_object, rect, frame_idx))
        
        frames.append(sly.Frame(frame_idx, frame_figures))

    objects = list(video_objects.values())
    
    return VideoAnnotation(
            img_size=frame_shape,
            frames_count=len(predictions),
            objects=sly.VideoObjectCollection(objects),
            frames=sly.FrameCollection(frames)
        )
    