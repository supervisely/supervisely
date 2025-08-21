def visualize(predictions, output_path, **kwargs)
    pass

def visualize_video_annotation(video_annotation, output_path, **kwargs):
    pass


##
# Visualize Supervisely tracking annotations on video
import json
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
import os
from collections import defaultdict



def load_supervisely_annotation(json_path):
    """Load Supervisely video annotation from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_track_color(track_id, seed=42):
    """Generate consistent color for each track ID."""
    np.random.seed(track_id + seed)
    # Generate bright colors
    color = tuple(np.random.randint(60, 255, 3).tolist())
    return color


def load_video_source(config):
    """Load video source (either video file or image directory)."""
    if config['input_type'] == 'video':
        cap = cv2.VideoCapture(config['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        return frames, fps, frame_count
    
    elif config['input_type'] == 'images':
        image_dir = Path(config['images_dir'])
        image_files = sorted([f for f in image_dir.iterdir() 
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        frames = []
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not load {img_path}")
                continue
            frames.append(frame)
        
        fps = config.get('output_fps', 10)  # Default FPS for image sequence
        frame_count = len(frames)
        
        return frames, fps, frame_count
    
    else:
        raise ValueError("input_type must be 'video' or 'images'")


def extract_tracks_data(sly_data):
    """Extract track information from Supervisely annotation."""
    tracks = defaultdict(list)  # track_id -> list of (frame_idx, bbox, class_title)
    
    # Create mapping from objectKey to track info
    objects_info = {}
    for i, obj in enumerate(sly_data['objects']):
        objects_info[obj['key']] = {
            'track_id': i + 1,
            'class_title': obj['classTitle']
        }
    
    # Process frames
    for frame in sly_data['frames']:
        frame_idx = frame['index']
        
        for figure in frame['figures']:
            if figure['geometryType'] != 'rectangle':
                continue
                
            object_key = figure['objectKey']
            track_info = objects_info[object_key]
            track_id = track_info['track_id']
            class_title = track_info['class_title']
            
            # Extract bounding box
            exterior_points = figure['geometry']['points']['exterior']
            x1, y1 = exterior_points[0]
            x2, y2 = exterior_points[1]
            
            # Normalize coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            tracks[track_id].append({
                'frame_idx': frame_idx,
                'bbox': bbox,
                'class_title': class_title
            })
    
    return tracks


def draw_trajectory(img, track_points, color, max_points=30):
    """Draw trajectory line for a track."""
    if len(track_points) < 2:
        return
    
    # Limit trajectory length
    points = track_points[-max_points:]
    
    # Draw trajectory lines
    for i in range(1, len(points)):
        cv2.line(img, points[i-1], points[i], color, 2)
    
    # Draw points
    for point in points[:-1]:
        cv2.circle(img, point, 3, color, -1)


def visualize_frame(frame, tracks, frame_idx, config):
    """Draw tracking annotations on a single frame."""
    img = frame.copy()
    
    # Track center points for trajectory
    trajectory_points = defaultdict(list)
    
    for track_id, track_data in tracks.items():
        # Find detections for current frame
        current_detections = [d for d in track_data if d['frame_idx'] == frame_idx]
        
        if not current_detections:
            continue
        
        detection = current_detections[0]  # Should be only one per frame
        bbox = detection['bbox']
        class_title = detection['class_title']
        color = get_track_color(track_id)
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, config.get('box_thickness', 2))
        
        # Draw track ID and class
        if config.get('show_track_ids', True):
            label = f"ID:{track_id}"
            if config.get('show_class_names', True):
                label += f" ({class_title})"
            
            # Calculate text position
            text_y = y1 - 10 if y1 > 30 else y2 + 25
            
            # Draw text background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 
                config.get('text_scale', 0.6), config.get('text_thickness', 2)
            )
            cv2.rectangle(img, (x1, text_y - text_h - 5), (x1 + text_w, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(
                img, label, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, config.get('text_scale', 0.6),
                (255, 255, 255), config.get('text_thickness', 2), cv2.LINE_AA
            )
        
        # Store center point for trajectory
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        trajectory_points[track_id].append((center_x, center_y))
    
    # Draw trajectories if enabled
    if config.get('show_trajectories', False):
        # Collect all trajectory points up to current frame
        all_trajectory_points = defaultdict(list)
        
        for track_id, track_data in tracks.items():
            for detection in track_data:
                if detection['frame_idx'] <= frame_idx:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    all_trajectory_points[track_id].append((center_x, center_y))
        
        # Draw trajectories
        for track_id, points in all_trajectory_points.items():
            if len(points) > 1:
                color = get_track_color(track_id)
                draw_trajectory(img, points, color, config.get('trajectory_length', 30))
    
    return img


def main():
    with open("visualize_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load annotation
    sly_data = load_supervisely_annotation(config['annotation_path'])
    print(f"Loaded annotation with {len(sly_data['objects'])} objects and {len(sly_data['frames'])} frames")
    
    # Load video source
    frames, fps, frame_count = load_video_source(config)
    print(f"Loaded {len(frames)} frames with FPS: {fps}")
    
    # Extract tracking data
    tracks = extract_tracks_data(sly_data)
    print(f"Extracted {len(tracks)} tracks")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*config.get('codec', 'XVID'))
        output_path = output_dir / config['output_filename']
        
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (w, h)
        )
        
        print(f"Processing {len(frames)} frames...")
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            if frame_idx > len(tracks):
                print(f"Warning: Frame index {frame_idx} exceeds available tracks, stopping early.")
                break
            # Visualize tracking on frame
            vis_frame = visualize_frame(frame, tracks, frame_idx, config)
            
            # Add frame number if enabled
            if config.get('show_frame_number', False):
                cv2.putText(
                    vis_frame, f"Frame: {frame_idx + 1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA
                )
            
            # Write frame
            video_writer.write(vis_frame)
            
            # Progress indicator
            if (frame_idx + 1) % 100 == 0:
                print(f"Processed {frame_idx + 1}/{len(frames)} frames")
        
        video_writer.release()
        print(f"Visualization saved to: {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()