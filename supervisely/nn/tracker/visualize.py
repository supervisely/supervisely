from typing import Union, Dict, List, Tuple
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import supervisely as sly
from supervisely import logger
from supervisely.nn.model.prediction import Prediction
from supervisely import VideoAnnotation
from .utils import predictions_to_video_annotation

@dataclass
class VisualizationConfig:
    """Configuration for video tracking visualization."""
    # Visual appearance
    show_labels: bool = True
    show_classes: bool = True
    show_trajectories: bool = True
    show_frame_number: bool = True
    
    # Style settings
    box_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 2
    trajectory_length: int = 30
    
    # Output settings
    codec: str = 'XVID'
    output_fps: float = 30.0
    
    def update(self, **kwargs) -> 'VisualizationConfig':
        """Update config with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def get_track_color(track_id: int, seed: int = 42) -> Tuple[int, int, int]:
    """Generate consistent color for each track ID."""
    np.random.seed(track_id + seed)
    return tuple(np.random.randint(60, 255, 3).tolist())


def load_video_frames(source: Union[str, Path], output_fps: float) -> Tuple[List[np.ndarray], float]:
    """
    Load frames from video file or image directory.
    
    Args:
        source: Path to video file or directory with images
        
    Returns:
        Tuple of (frames list, fps)
        
    Raises:
        ValueError: If source is invalid or no frames found
    """
    source = Path(source)
    
    if source.is_file():  # Video file
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {source}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {source}")
        return frames, fps
    
    elif source.is_dir():  # Image directory
        # Support common image extensions (including uppercase)
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(source.glob(f'*{ext}'))
        
        image_files = sorted(image_files)
        if not image_files:
            raise ValueError(f"No image files found in directory: {source}")
        
        frames = []
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Could not read image: {img_path}")
        
        if not frames:
            raise ValueError(f"No valid images loaded from: {source}")
        
        return frames, output_fps
    
    else:
        raise ValueError(f"Source must be a video file or image directory, got: {source}")


def extract_tracks_from_sly_annotation(annotation: Union[VideoAnnotation, dict]) -> Dict[int, List[Tuple[int, Tuple[int, int, int, int], str]]]:
    """
    Extract tracks from Supervisely VideoAnnotation or dict.
    
    Args:
        annotation: Supervisely VideoAnnotation object or dict
        
    Returns:
        Dict mapping frame_idx to list of (track_id, bbox, class_name) tuples
        
    Raises:
        TypeError: If annotation type is not supported
        ValueError: If annotation structure is invalid
    """
    # Convert VideoAnnotation to dict if needed
    if isinstance(annotation, VideoAnnotation):
        annotation_dict = annotation.to_json()
    elif isinstance(annotation, dict):
        annotation_dict = annotation
    else:
        raise TypeError(f"Annotation must be VideoAnnotation or dict, got {type(annotation)}")
    

    
    # Map object keys to track info
    objects = {}
    for i, obj in enumerate(annotation_dict['objects']):
        if 'key' not in obj or 'classTitle' not in obj:
            continue
        objects[obj['key']] = (i, obj['classTitle'])
    
    # Group detections by frame
    frames_data = defaultdict(list)
    
    for frame in annotation_dict['frames']:
        frame_idx = frame['index']
        for figure in frame['figures']:
            # Validate figure structure
            if figure['geometryType'] != 'rectangle':
                continue
                
            object_key = figure['objectKey']
            if object_key not in objects:
                continue
                
            track_id, class_name = objects[object_key]
            
            # Extract bbox using Supervisely format
            points = figure['geometry']['points']['exterior']                
            x1, y1 = points[0]
            x2, y2 = points[1]
        
            # Ensure proper bbox format (x1, y1, x2, y2) with x1 < x2, y1 < y2
            bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            frames_data[frame_idx].append((track_id, bbox, class_name))

    return frames_data


def draw_detection(img: np.ndarray, track_id: int, bbox: Tuple[int, int, int, int], 
                  class_name: str, config: VisualizationConfig) -> Tuple[int, int]:
    """
    Draw single detection with track ID and class.
    
    Returns:
        Center point coordinates (cx, cy)
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = get_track_color(track_id)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, config.box_thickness)
    
    # Draw label
    if config.show_labels:
        label = f"ID:{track_id}"
        if config.show_classes:
            label += f" ({class_name})"
        
        # Position label above bbox if there's space, otherwise below
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        
        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.text_scale, config.text_thickness
        )
        
        # Draw text background
        cv2.rectangle(img, (x1, label_y - text_h - 5), 
                     (x1 + text_w, label_y + 5), color, -1)
        
        # Draw text
        cv2.putText(img, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, config.text_scale, 
                   (255, 255, 255), config.text_thickness, cv2.LINE_AA)
    
    # Return center point for trajectory
    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_trajectories(img: np.ndarray, track_centers: Dict[int, List[Tuple[int, int]]], 
                     config: VisualizationConfig) -> None:
    """Draw trajectory lines for all tracks."""
    if not config.show_trajectories:
        return
    
    for track_id, centers in track_centers.items():
        if len(centers) < 2:
            continue
            
        color = get_track_color(track_id)
        # Limit trajectory length to avoid clutter
        points = centers[-config.trajectory_length:]
        
        # Draw trajectory line
        for i in range(1, len(points)):
            cv2.line(img, points[i-1], points[i], color, 2)
        
        # Draw trajectory points (except the last one which is current position)
        for point in points[:-1]:
            cv2.circle(img, point, 3, color, -1)


def visualize_video_annotation(annotation: Union[VideoAnnotation, dict], 
                              source: Union[str, Path], 
                              output_path: Union[str, Path], 
                              **kwargs) -> None:
    """
    Visualize tracking annotations on video.
    
    Args:
        annotation: Supervisely VideoAnnotation object or dict
        source: Path to video file or image directory  
        output_path: Path for output video
        **kwargs: Additional configuration parameters
        
    Raises:
        TypeError: If annotation type is not supported
        ValueError: If source/annotation is invalid
    """
    # Create configuration with defaults and user overrides
    config = VisualizationConfig().update(**kwargs)
    
    # Load video frames
    frames, fps = load_video_frames(source, config.output_fps)
    logger.info(f"Loaded {len(frames)} frames from {source} at {fps:.1f} FPS")
    
    # Extract tracking data 
    tracks_by_frame = extract_tracks_from_sly_annotation(annotation)
    logger.info(f"Extracted {len(tracks_by_frame)} tracks from annotation")
    
    if not frames:
        raise ValueError("No frames loaded from source")
    
    # Setup video writer
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer for {output_path}")
    
    # Track centers for trajectory visualization
    track_centers = defaultdict(list)
    
    try:
        for frame_idx, frame in enumerate(frames):
            img = frame.copy()
            
            # Draw detections for current frame
            if frame_idx in tracks_by_frame:
                for track_id, bbox, class_name in tracks_by_frame[frame_idx]:
                    center = draw_detection(img, track_id, bbox, class_name, config)
                    track_centers[track_id].append(center)
            
            # Draw trajectories
            draw_trajectories(img, track_centers, config)
            
            # Add frame number if requested
            if config.show_frame_number:
                cv2.putText(img, f"Frame: {frame_idx + 1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            writer.write(img)
            
            # Progress update every 100 frames
            if (frame_idx + 1) % 100 == 0:
                logger.info(f"Processed {frame_idx + 1}/{len(frames)} frames")
    
    finally:
        writer.release()
        
    logger.info(f"Video saved to {output_path}")



def visualize(predictions: Union[VideoAnnotation, dict, Prediction], 
              source: Union[str, Path], 
              output_path: Union[str, Path], 
              **kwargs) -> None:
    """
    Generic visualization function for different input types.
    
    Args:
        predictions: VideoAnnotation, dict, or Prediction object
        source: Path to video file or image directory
        output_path: Path for output video  
        **kwargs: Additional configuration parameters
    """
    annotation = predictions_to_video_annotation(predictions)
    visualize_video_annotation(annotation, source, output_path, **kwargs)