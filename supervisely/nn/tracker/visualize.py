import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import ffmpeg
import numpy as np

import supervisely as sly
from supervisely import VideoAnnotation, logger
from supervisely.nn.model.prediction import Prediction
from supervisely.nn.tracker.utils import predictions_to_video_annotation


class TrackingVisualizer:

    def __init__(
        self,
        show_labels: bool = True,
        show_classes: bool = True,
        show_trajectories: bool = True,
        show_frame_number: bool = False,
        box_thickness: int = 2,
        text_scale: float = 0.6,
        text_thickness: int = 2,
        trajectory_length: int = 30,
        codec: str = "mp4",
        output_fps: float = 30.0,
        colorize_tracks: bool = True,
        trajectory_thickness: int = 2,
    ):
        """
        Initialize the visualizer with configuration.

        Args:
            show_labels: Whether to show track IDs.
            show_classes: Whether to show class names.
            show_trajectories: Whether to draw trajectories.
            show_frame_number: Whether to overlay frame number.
            box_thickness: Thickness of bounding boxes.
            text_scale: Scale of label text.
            text_thickness: Thickness of label text.
            trajectory_length: How many points to keep in trajectory.
            codec: Output video codec.
            output_fps: Output video framerate.
            colorize_tracks (bool, default=True): if True, ignore colors from project meta and generate new colors for each tracked object; if False, try to use colors from project meta when possible.
        """
        # Visualization settings
        self.show_labels = show_labels
        self.show_classes = show_classes
        self.show_trajectories = show_trajectories
        self.show_frame_number = show_frame_number

        # Style settings
        self.box_thickness = box_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.trajectory_length = trajectory_length
        self.trajectory_thickness = trajectory_thickness
        self.colorize_tracks = colorize_tracks

        # Output settings
        self.codec = codec
        self.output_fps = output_fps

        # Internal state
        self.annotation = None
        self.tracks_by_frame = {}
        self.track_centers = defaultdict(list)
        self.track_colors = {}
        self.color_palette = self._generate_color_palette()
        self._temp_dir = None

    def _generate_color_palette(self, num_colors: int = 100) -> List[Tuple[int, int, int]]:
        """
        Generate bright, distinct color palette for track visualization.
        Uses HSV space with random hue and fixed high saturation/value.
        """
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            hue = np.random.randint(0, 180)
            saturation = 200 + np.random.randint(55)
            value = 200 + np.random.randint(55)

            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))
        return colors

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID from palette."""
        return self.color_palette[track_id % len(self.color_palette)]

    def _get_video_info(self, video_path: Path) -> Tuple[int, int, float, int]:
        """
        Get video metadata using ffmpeg.
        
        Returns:
            Tuple of (width, height, fps, total_frames)
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)

            if video_stream is None:
                raise ValueError(f"No video stream found in: {video_path}")

            width = int(video_stream['width'])
            height = int(video_stream['height'])

            # Extract FPS
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)

            # Get total frames
            total_frames = int(video_stream.get('nb_frames', 0))
            if total_frames == 0:
                # Fallback: estimate from duration and fps
                duration = float(video_stream.get('duration', 0))
                total_frames = int(duration * fps) if duration > 0 else 0

            return width, height, fps, total_frames

        except Exception as e:
            raise ValueError(f"Could not read video metadata {video_path}: {str(e)}")

    def _create_frame_iterator(self, source: Union[str, Path]) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Create iterator that yields (frame_index, frame) tuples.
        
        Args:
            source: Path to video file or directory with frame images
            
        Yields:
            Tuple of (frame_index, frame_array)
        """
        source = Path(source)

        if source.is_file():
            yield from self._iterate_video_frames(source)
        elif source.is_dir():
            yield from self._iterate_directory_frames(source)
        else:
            raise ValueError(f"Source must be a video file or directory, got: {source}")

    def _iterate_video_frames(self, video_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate through video frames using ffmpeg."""
        width, height, fps, total_frames = self._get_video_info(video_path)

        # Store video info for later use
        self.source_fps = fps
        self.frame_size = (width, height)

        process = (
            ffmpeg
            .input(str(video_path))
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
            .run_async(pipe_stdout=True, pipe_stderr=False)
        )

        try:
            frame_size_bytes = width * height * 3
            frame_idx = 0

            while True:
                frame_data = process.stdout.read(frame_size_bytes)
                if len(frame_data) != frame_size_bytes:
                    break

                frame = np.frombuffer(frame_data, np.uint8).reshape([height, width, 3])
                yield frame_idx, frame
                frame_idx += 1

        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}", exc_info=True)

        finally:
            process.stdout.close()
            if process.stderr:
                process.stderr.close()
            process.wait()

    def _iterate_directory_frames(self, frames_dir: Path) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate through image frames in directory."""
        if not frames_dir.is_dir():
            raise ValueError(f"Directory does not exist: {frames_dir}")

        # Support common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(frames_dir.glob(f'*{ext}'))
            image_files.extend(frames_dir.glob(f'*{ext.upper()}'))

        image_files = sorted(image_files)
        if not image_files:
            raise ValueError(f"No image files found in directory: {frames_dir}")

        # Set fps from config for image sequences
        self.source_fps = self.output_fps

        for frame_idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                if frame_idx == 0:
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)
                yield frame_idx, frame
            else:
                logger.warning(f"Could not read image: {img_path}")

    def _extract_tracks_from_annotation(self) -> None:
        """
        Extract tracking data from Supervisely VideoAnnotation.
        
        Populates self.tracks_by_frame with frame-indexed tracking data.
        """
        self.tracks_by_frame = defaultdict(list)
        self.track_colors = {}

        # Map object keys to track info
        objects = {}
        for i, obj in enumerate(self.annotation.objects):
            objects[obj.key] = (i, obj.obj_class.name)

        # Extract tracks from frames
        for frame in self.annotation.frames:
            frame_idx = frame.index
            for figure in frame.figures:
                if figure.geometry.geometry_name() != 'rectangle':
                    continue

                object_key = figure.parent_object.key
                if object_key not in objects:
                    continue

                track_id, class_name = objects[object_key]

                # Extract bbox coordinates
                rect = figure.geometry
                bbox = (rect.left, rect.top, rect.right, rect.bottom)

                if track_id not in self.track_colors:
                    if self.colorize_tracks:
                        # auto-color override everything
                        color = self._get_track_color(track_id)
                    else:
                        # try to use annotation color
                        color = figure.video_object.obj_class.color
                        if color:
                            # convert rgb → bgr
                            color = color[::-1]
                        else:
                            # fallback to auto-color if annotation missing
                            color = self._get_track_color(track_id)

                    self.track_colors[track_id] = color

                self.tracks_by_frame[frame_idx].append((track_id, bbox, class_name))

        logger.info(f"Extracted tracks from {len(self.tracks_by_frame)} frames")

    def _draw_detection(self, img: np.ndarray, track_id: int, bbox: Tuple[int, int, int, int], 
                    class_name: str) -> Optional[Tuple[int, int]]:
        """
        Draw single detection with track ID and class label.
        Returns the center point of the bbox for trajectory drawing.
        """
        x1, y1, x2, y2 = map(int, bbox)

        if x2 <= x1 or y2 <= y1:
            return None

        color = self.track_colors[track_id]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.box_thickness)

        # Draw label if enabled
        if self.show_labels:
            label = f"ID:{track_id}"
            if self.show_classes:
                label += f" ({class_name})"

            label_y = y1 - 10 if y1 > 30 else y2 + 25
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )

            cv2.rectangle(img, (x1, label_y - text_h - 5), 
                        (x1 + text_w, label_y + 5), color, -1)
            cv2.putText(img, label, (x1, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                    (255, 255, 255), self.text_thickness, cv2.LINE_AA)

        # Return center point for trajectory
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _draw_trajectories(self, img: np.ndarray) -> None:
        """Draw trajectory lines for all tracks, filtering out big jumps."""
        if not self.show_trajectories:
            return

        max_jump = 200  

        for track_id, centers in self.track_centers.items():
            if len(centers) < 2:
                continue

            color = self.track_colors[track_id]
            points = centers[-self.trajectory_length:]

            for i in range(1, len(points)):
                p1, p2 = points[i - 1], points[i]
                if p1 is None or p2 is None:
                    continue

                if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) > max_jump:
                    continue
                cv2.line(img, p1, p2, color, self.trajectory_thickness)
                cv2.circle(img, p1, 3, color, -1)

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process single frame: add annotations and return processed frame.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Annotated frame
        """
        img = frame.copy()
        active_ids = set()
        # Draw detections for current frame
        if frame_idx in self.tracks_by_frame:
            for track_id, bbox, class_name in self.tracks_by_frame[frame_idx]:
                center = self._draw_detection(img, track_id, bbox, class_name)
                self.track_centers[track_id].append(center)
                if len(self.track_centers[track_id]) > self.trajectory_length:
                    self.track_centers[track_id].pop(0)
                active_ids.add(track_id)

        for tid in self.track_centers.keys():
            if tid not in active_ids:
                self.track_centers[tid].append(None)
                if len(self.track_centers[tid]) > self.trajectory_length:
                    self.track_centers[tid].pop(0)

        # Draw trajectories
        self._draw_trajectories(img)

        # Add frame number if requested
        if self.show_frame_number:
            cv2.putText(img, f"Frame: {frame_idx + 1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return img

    def _save_processed_frame(self, frame: np.ndarray, frame_idx: int) -> str:
        """
        Save processed frame to temporary directory.
        
        Args:
            frame: Processed frame
            frame_idx: Frame index
            
        Returns:
            Path to saved frame
        """
        frame_path = self._temp_dir / f"frame_{frame_idx:08d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(frame_path)

    def _create_video_from_frames(self, output_path: Union[str, Path]) -> None:
        """
        Create final video from processed frames using ffmpeg.
        
        Args:
            output_path: Path for output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create video from frame sequence
        input_pattern = str(self._temp_dir / "frame_%08d.jpg")

        try:
            (
                ffmpeg
                .input(input_pattern, pattern_type='sequence', framerate=self.source_fps)
                .output(str(output_path), vcodec='libx264', pix_fmt='yuv420p', crf=18)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Video saved to {output_path}")

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
            raise ValueError(f"Failed to create video: {error_msg}")

    def _cleanup_temp_directory(self) -> None:
        """Clean up temporary directory and all its contents."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def visualize_video_annotation(self, annotation: VideoAnnotation, 
                                  source: Union[str, Path], 
                                  output_path: Union[str, Path]) -> None:
        """
        Visualize tracking annotations on video using streaming approach.
        
        Args:
            annotation: Supervisely VideoAnnotation object with tracking data
            source: Path to video file or directory containing frame images
            output_path: Path for output video file
            
        Raises:
            TypeError: If annotation is not VideoAnnotation
            ValueError: If source is invalid or annotation is empty
        """
        if not isinstance(annotation, VideoAnnotation):
            raise TypeError(f"Annotation must be VideoAnnotation, got {type(annotation)}")

        # Store annotation
        self.annotation = annotation

        # Create temporary directory for processed frames
        self._temp_dir = Path(tempfile.mkdtemp(prefix="video_viz_"))

        try:
            # Extract tracking data
            self._extract_tracks_from_annotation()

            if not self.tracks_by_frame:
                logger.warning("No tracking data found in annotation")

            # Reset trajectory tracking
            self.track_centers = defaultdict(list)

            # Process frames one by one
            frame_count = 0
            for frame_idx, frame in self._create_frame_iterator(source):
                # Process frame
                processed_frame = self._process_single_frame(frame, frame_idx)

                # Save processed frame
                self._save_processed_frame(processed_frame, frame_idx)

                frame_count += 1

                # Progress logging
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

            logger.info(f"Finished processing {frame_count} frames")

            # Create final video from saved frames
            self._create_video_from_frames(output_path)

        finally:
            # Always cleanup temporary files
            self._cleanup_temp_directory()

    def __del__(self):
        """Cleanup temporary directory on object destruction."""
        self._cleanup_temp_directory()


def visualize(
    predictions: Union[VideoAnnotation, List[Prediction]], 
    source: Union[str, Path], 
    output_path: Union[str, Path],
    show_labels: bool = True,
    show_classes: bool = True,
    show_trajectories: bool = True,
    box_thickness: int = 2,
    colorize_tracks: bool = True,
    **kwargs
) -> None:
    """
    Visualize tracking results from either VideoAnnotation or list of Prediction.

    Args:
        predictions (supervisely.VideoAnnotation | List[Prediction]): Tracking data to render; either a Supervisely VideoAnnotation or a list of Prediction objects.
        source (str | Path): Path to an input video file or a directory of sequential frames (e.g., frame_000001.jpg).
        output_path (str | Path): Path to the output video file to be created.
        show_labels (bool, default=True): Draw per-object labels (track IDs).
        show_classes (bool, default=True): Draw class names for each object.
        show_trajectories (bool, default=True): Render object trajectories across frames.
        box_thickness (int, default=2): Bounding-box line thickness in pixels.
        colorize_tracks (bool, default=True): if True, ignore colors from project meta and generate new colors for each tracked object; if False, try to use colors from project meta when possible.    
        """
    visualizer = TrackingVisualizer(
        show_labels=show_labels, 
        show_classes=show_classes, 
        show_trajectories=show_trajectories,
        box_thickness=box_thickness,
        colorize_tracks=colorize_tracks,
        **kwargs
    )

    if isinstance(predictions, VideoAnnotation):
        visualizer.visualize_video_annotation(predictions, source, output_path)
    elif isinstance(predictions, list):
        predictions = predictions_to_video_annotation(predictions)
        visualizer.visualize_video_annotation(predictions, source, output_path)
    else:
        raise TypeError(f"Predictions must be VideoAnnotation or list of Prediction, got {type(predictions)}")
