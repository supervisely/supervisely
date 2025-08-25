import numpy as np
import tempfile
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import supervisely as sly
from supervisely import logger
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.nn.tracker.utils import video_annotation_to_mot

try:
    import motmetrics as mm
    import pandas as pd
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False
    logger.warning("motmetrics not available. Install with: pip install motmetrics")

try:
    import trackeval
    TRACKEVAL_AVAILABLE = True
except ImportError:
    TRACKEVAL_AVAILABLE = False
    logger.warning("trackeval not available. Install with: pip install git+https://github.com/JonathonLuiten/TrackEval.git")


class TrackingEvaluator:
    """
    Evaluator for video tracking metrics including MOTA, MOTP, IDF1, and HOTA.
    
    Converts Supervisely VideoAnnotations to MOT format and uses standard
    evaluation libraries for accurate metric computation.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize tracking evaluator.
        
        Args:
            iou_threshold: IoU threshold for matching detections (0.0 to 1.0)
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {iou_threshold}")
        
        self.iou_threshold = iou_threshold
    
    def evaluate(
        self, 
        gt_annotation: VideoAnnotation, 
        pred_annotation: VideoAnnotation
    ) -> Dict[str, Union[float, int]]:
        """
        Evaluate tracking performance between ground truth and predictions.
        
        Args:
            gt_annotation: Ground truth video annotation
            pred_annotation: Predicted video annotation
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(gt_annotation, pred_annotation)
        
        logger.info(f"Starting evaluation with IoU threshold: {self.iou_threshold}")
        
        # Convert annotations to MOT format
        gt_mot_lines = video_annotation_to_mot(gt_annotation)
        pred_mot_lines = video_annotation_to_mot(pred_annotation)
        
        if not gt_mot_lines:
            raise RuntimeError("No ground truth detections found")
        if not pred_mot_lines:
            raise RuntimeError("No prediction detections found")
        
        logger.info(f"Converted to MOT format: {len(gt_mot_lines)} GT, {len(pred_mot_lines)} pred detections")
        
        # Compute metrics using MOT format
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save MOT files
            gt_file = self._save_mot_file(gt_mot_lines, temp_dir, "gt.txt")
            pred_file = self._save_mot_file(pred_mot_lines, temp_dir, "pred.txt")
            
            # Compute all metrics
            basic_metrics = self._compute_basic_metrics(gt_mot_lines, pred_mot_lines)
            mot_metrics = self._compute_mot_metrics(gt_file, pred_file)
            hota_metrics = self._compute_hota_metrics(gt_file, pred_file, temp_dir)
        
        # Combine results
        results = {
            # Basic detection metrics
            'precision': basic_metrics['precision'],
            'recall': basic_metrics['recall'],
            'f1': basic_metrics['f1'],
            'avg_iou': basic_metrics['avg_iou'],
            
            # MOT metrics
            'mota': mot_metrics['mota'],
            'motp': mot_metrics['motp'],
            'idf1': mot_metrics['idf1'],
            'id_switches': mot_metrics['id_switches'],
            'fragmentations': mot_metrics['fragmentations'],
            'num_misses': mot_metrics['num_misses'],
            'num_false_positives': mot_metrics['num_false_positives'],
            
            # HOTA metrics  
            'hota': hota_metrics['hota'],
            'deta': hota_metrics['deta'],
            'assa': hota_metrics['assa'],
            
            # Count metrics
            'true_positives': basic_metrics['tp'],
            'false_positives': basic_metrics['fp'],
            'false_negatives': basic_metrics['fn'],
            'total_gt_objects': basic_metrics['total_gt'],
            'total_pred_objects': basic_metrics['total_pred'],
            
            # Config
            'iou_threshold': self.iou_threshold
        }
        
        logger.info(
            f"Evaluation complete - MOTA: {results['mota']:.3f}, "
            f"HOTA: {results['hota']:.3f}, Precision: {results['precision']:.3f}"
        )
        
        return results
    
    def _validate_inputs(self, gt_annotation: VideoAnnotation, pred_annotation: VideoAnnotation):
        """Validate input annotations."""
        if not isinstance(gt_annotation, VideoAnnotation):
            raise TypeError(f"Ground truth must be VideoAnnotation, got {type(gt_annotation)}")
        
        if not isinstance(pred_annotation, VideoAnnotation):
            raise TypeError(f"Predictions must be VideoAnnotation, got {type(pred_annotation)}")
        
        if gt_annotation.frames_count == 0:
            raise ValueError("Ground truth annotation is empty")
        
        if pred_annotation.frames_count == 0:
            raise ValueError("Prediction annotation is empty")
    
    def _save_mot_file(self, mot_lines: List[str], temp_dir: str, filename: str) -> str:
        """Save MOT format lines to file."""
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            for line in mot_lines:
                f.write(line + '\n')
        return file_path
    
    def _parse_mot_file(self, file_path: str) -> Dict[int, List[Dict]]:
        """Parse MOT file into tracks by frame."""
        tracks_by_frame = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split(',')
                    if len(parts) < 9:
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    left = float(parts[2])
                    top = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    confidence = float(parts[6])
                    class_id = int(parts[7])
                    visibility = float(parts[8])
                    
                    frame_idx = frame_id - 1  # Convert to 0-based
                    
                    if frame_idx not in tracks_by_frame:
                        tracks_by_frame[frame_idx] = []
                    
                    tracks_by_frame[frame_idx].append({
                        'track_id': track_id,
                        'bbox': [left, top, left + width, top + height],  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'class_id': class_id,
                        'visibility': visibility
                    })
                
                except (ValueError, IndexError):
                    continue
        
        return tracks_by_frame
    
    def _compute_basic_metrics(self, gt_mot_lines: List[str], pred_mot_lines: List[str]) -> Dict[str, float]:
        """Compute basic detection metrics from MOT format data."""
        # Parse MOT lines to temporary files then parse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gt_tmp:
            gt_tmp.write('\n'.join(gt_mot_lines))
            gt_tmp_path = gt_tmp.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pred_tmp:
            pred_tmp.write('\n'.join(pred_mot_lines))
            pred_tmp_path = pred_tmp.name
        
        try:
            gt_tracks = self._parse_mot_file(gt_tmp_path)
            pred_tracks = self._parse_mot_file(pred_tmp_path)
        finally:
            # Cleanup temp files
            os.unlink(gt_tmp_path)
            os.unlink(pred_tmp_path)
        
        tp = fp = fn = 0
        total_iou = 0.0
        iou_count = 0
        
        # Get all frames
        all_frames = set(list(gt_tracks.keys()) + list(pred_tracks.keys()))
        
        for frame_idx in all_frames:
            gt_objects = gt_tracks.get(frame_idx, [])
            pred_objects = pred_tracks.get(frame_idx, [])
            
            if len(gt_objects) > 0 and len(pred_objects) > 0:
                # Calculate IoU matrix
                gt_boxes = np.array([obj['bbox'] for obj in gt_objects])
                pred_boxes = np.array([obj['bbox'] for obj in pred_objects])
                
                iou_matrix = self._calculate_iou_matrix(gt_boxes, pred_boxes)
                
                # Find matches
                matches = self._find_matches(iou_matrix, self.iou_threshold)
                
                frame_tp = len(matches)
                frame_fp = len(pred_objects) - frame_tp
                frame_fn = len(gt_objects) - frame_tp
                
                tp += frame_tp
                fp += frame_fp
                fn += frame_fn
                
                # Calculate average IoU for matched pairs
                for gt_idx, pred_idx in matches:
                    total_iou += iou_matrix[gt_idx, pred_idx]
                    iou_count += 1
            else:
                fp += len(pred_objects)
                fn += len(gt_objects)
        
        # Calculate final metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
        
        total_gt = sum(len(tracks) for tracks in gt_tracks.values())
        total_pred = sum(len(tracks) for tracks in pred_tracks.values())
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': total_gt,
            'total_pred': total_pred
        }
    
    def _parse_mot_lines_to_tracks(self, mot_lines: List[str]) -> Dict[int, List[Dict]]:
        """Parse MOT lines into tracks by frame."""
        tracks_by_frame = {}
        
        for line in mot_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                confidence = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])
                
                frame_idx = frame_id - 1  # Convert to 0-based
                
                if frame_idx not in tracks_by_frame:
                    tracks_by_frame[frame_idx] = []
                
                tracks_by_frame[frame_idx].append({
                    'track_id': track_id,
                    'bbox': [left, top, left + width, top + height],  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'visibility': visibility
                })
            
            except (ValueError, IndexError):
                continue
        
        return tracks_by_frame
    
    def _compute_mot_metrics(self, gt_file: str, pred_file: str) -> Dict[str, Union[float, int]]:
        """Compute MOT metrics using motmetrics library."""
        if not MOTMETRICS_AVAILABLE:
            logger.warning("motmetrics not available - returning zero MOT metrics")
            return {
                'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                'id_switches': 0, 'fragmentations': 0,
                'num_misses': 0, 'num_false_positives': 0
            }
        
        try:
            # Parse MOT files
            gt_tracks = self._parse_mot_file(gt_file)
            pred_tracks = self._parse_mot_file(pred_file)
            
            # Create MOT accumulator
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Process each frame
            all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
            
            for frame_idx in all_frames:
                gt_objects = gt_tracks.get(frame_idx, [])
                pred_objects = pred_tracks.get(frame_idx, [])
                
                # Extract track IDs
                gt_ids = [obj['track_id'] for obj in gt_objects]
                pred_ids = [obj['track_id'] for obj in pred_objects]
                
                # Calculate distance matrix (1 - IoU)
                if len(gt_objects) > 0 and len(pred_objects) > 0:
                    gt_boxes = np.array([obj['bbox'] for obj in gt_objects])
                    pred_boxes = np.array([obj['bbox'] for obj in pred_objects])
                    
                    iou_matrix = self._calculate_iou_matrix(gt_boxes, pred_boxes)
                    distance_matrix = 1.0 - iou_matrix
                    
                    # Set infinite distance for matches below threshold
                    distance_matrix[iou_matrix < self.iou_threshold] = np.inf
                else:
                    distance_matrix = np.empty((len(gt_objects), len(pred_objects)))
                    distance_matrix.fill(np.inf)
                
                # Update accumulator
                acc.update(gt_ids, pred_ids, distance_matrix)
            
            # Compute metrics
            mh = mm.metrics.create()
            summary = mh.compute(
                acc,
                metrics=['mota', 'motp', 'idf1', 'num_switches', 'num_fragmentations', 
                        'num_misses', 'num_false_positives'],
                name='tracking'
            )
            
            # Extract results safely
            def safe_extract(metric_name, default_value=0.0, as_int=False):
                if summary.empty or metric_name not in summary.columns:
                    return int(default_value) if as_int else default_value
                
                value = summary[metric_name].iloc[0]
                if pd.isna(value):
                    return int(default_value) if as_int else default_value
                
                return int(value) if as_int else float(value)
            
            return {
                'mota': safe_extract('mota', 0.0),
                'motp': safe_extract('motp', 0.0),
                'idf1': safe_extract('idf1', 0.0),
                'id_switches': safe_extract('num_switches', 0, as_int=True),
                'fragmentations': safe_extract('num_fragmentations', 0, as_int=True),
                'num_misses': safe_extract('num_misses', 0, as_int=True),
                'num_false_positives': safe_extract('num_false_positives', 0, as_int=True)
            }
            
        except Exception as e:
            logger.error(f"Failed to compute MOT metrics: {e}")
            return {
                'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                'id_switches': 0, 'fragmentations': 0,
                'num_misses': 0, 'num_false_positives': 0
            }
    
    def _compute_hota_metrics(self, gt_file: str, pred_file: str, temp_dir: str) -> Dict[str, float]:
        """Compute HOTA metrics using TrackEval."""
        if not TRACKEVAL_AVAILABLE:
            logger.warning("trackeval not available - returning zero HOTA metrics")
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
        
        try:
            # Create TrackEval directory structure
            self._setup_trackeval_dirs(gt_file, pred_file, temp_dir)
            
            # Run TrackEval
            config = {
                'USE_PARALLEL': False,
                'NUM_PARALLEL_CORES': 1,
                'BREAK_ON_ERROR': True,
                'RETURN_ON_ERROR': False,
                'PRINT_RESULTS': False,
                'PRINT_CONFIG': False,
                'TIME_PROGRESS': False,
                'DISPLAY_LESS_PROGRESS': True,
                
                'GT_FOLDER': os.path.join(temp_dir, 'trackeval', 'gt'),
                'TRACKERS_FOLDER': os.path.join(temp_dir, 'trackeval', 'trackers'),
                'OUTPUT_FOLDER': os.path.join(temp_dir, 'trackeval', 'output'),
                'TRACKERS_TO_EVAL': ['tracker'],
                'DATASETS_TO_EVAL': ['seq'],
                'BENCHMARK': 'MOT17',
                'SPLIT_TO_EVAL': 'train',
                'METRICS': ['HOTA'],
                'THRESHOLD': self.iou_threshold
            }
            
            # Run evaluation
            evaluator = trackeval.Evaluator(config)
            output_res, _ = evaluator.evaluate()
            
            # Extract HOTA results
            if output_res and 'seq' in output_res:
                tracker_results = output_res['seq']['tracker']['COMBINED_SEQ']
                
                if 'HOTA' in tracker_results:
                    hota_data = tracker_results['HOTA']
                    return {
                        'hota': hota_data.get('HOTA', 0.0),
                        'deta': hota_data.get('DetA', 0.0),
                        'assa': hota_data.get('AssA', 0.0)
                    }
            
        except Exception as e:
            logger.debug(f"TrackEval failed: {e}")
        
        return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
    
    def _setup_trackeval_dirs(self, gt_file: str, pred_file: str, temp_dir: str):
        """Setup TrackEval directory structure."""
        # Create directories
        eval_dir = Path(temp_dir) / 'trackeval'
        gt_seq_dir = eval_dir / 'gt' / 'seq'
        gt_data_dir = gt_seq_dir / 'gt'
        tracker_dir = eval_dir / 'trackers' / 'tracker'
        seqmap_dir = eval_dir / 'gt' / 'seqmaps'
        
        gt_data_dir.mkdir(parents=True, exist_ok=True)
        tracker_dir.mkdir(parents=True, exist_ok=True)
        seqmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy MOT files
        import shutil
        shutil.copy2(gt_file, gt_data_dir / 'gt.txt')
        shutil.copy2(pred_file, tracker_dir / 'seq.txt')
        
        # Create seqinfo.ini
        gt_tracks = self._parse_mot_file(gt_file)
        pred_tracks = self._parse_mot_file(pred_file)
        
        max_frame = 0
        if gt_tracks:
            max_frame = max(max_frame, max(gt_tracks.keys()))
        if pred_tracks:
            max_frame = max(max_frame, max(pred_tracks.keys()))
        
        with open(gt_seq_dir / 'seqinfo.ini', 'w') as f:
            f.write("[Sequence]\n")
            f.write("name=seq\n")
            f.write(f"seqLength={max_frame + 1}\n")
            f.write("imWidth=1920\n")
            f.write("imHeight=1080\n")
            f.write("imExt=.jpg\n")
            f.write("imDir=img1\n")
            f.write("frameRate=30\n")
        
        # Create seqmap file
        with open(seqmap_dir / 'MOT17-train.txt', 'w') as f:
            f.write("name\n")
            f.write("seq\n")
    
    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between two sets of bounding boxes."""
        # boxes format: [x1, y1, x2, y2]
        areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        union = areas1[:, None] + areas2[None, :] - intersection
        
        # Calculate IoU
        iou = np.where(union > 0, intersection / union, 0)
        
        return iou
    
    def _find_matches(self, iou_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Find optimal matches using greedy assignment."""
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Get all candidates above threshold
        candidates = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] >= threshold:
                    candidates.append((iou_matrix[i, j], i, j))
        
        # Sort by IoU descending for greedy assignment
        candidates.sort(reverse=True)
        
        # Assign greedily
        for iou_val, gt_idx, pred_idx in candidates:
            if gt_idx not in used_gt and pred_idx not in used_pred:
                matches.append((gt_idx, pred_idx))
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
        
        return matches


def evaluate(
    gt_annotation: VideoAnnotation,
    pred_annotation: VideoAnnotation,
    iou_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    """
    Evaluate tracking performance between ground truth and predictions.
    
    This function provides a simple interface for tracking evaluation,
    similar to how visualize() works for visualization.
    
    Args:
        gt_annotation: Ground truth video annotation
        pred_annotation: Predicted video annotation  
        iou_threshold: IoU threshold for matching detections
        
    Returns:
        Dictionary containing all computed tracking metrics
        
    Usage:
        import supervisely as sly
        from supervisely.nn.tracker import evaluate
        
        # Load annotations  
        gt_ann = sly.VideoAnnotation.load_json_file("gt.json", project_meta)
        pred_ann = sly.VideoAnnotation.load_json_file("pred.json", project_meta)
        
        # Evaluate tracking
        metrics = evaluate(gt_ann, pred_ann, iou_threshold=0.5)
        
        print(f"MOTA: {metrics['mota']:.3f}")
        print(f"HOTA: {metrics['hota']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
    """
    evaluator = TrackingEvaluator(iou_threshold=iou_threshold)
    return evaluator.evaluate(gt_annotation, pred_annotation)

import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Union
from collections import defaultdict
import supervisely as sly
from supervisely import logger
from supervisely.video_annotation.video_annotation import VideoAnnotation

try:
    import motmetrics as mm
    import pandas as pd
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False

try:
    import trackeval
    TRACKEVAL_AVAILABLE = True
except ImportError:
    TRACKEVAL_AVAILABLE = False


def evaluate_mot_direct(
    gt_annotation: VideoAnnotation,
    pred_annotation: VideoAnnotation,
    iou_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    """
    Direct MOT evaluation without coordinate conversion through sly.Rectangle.
    Extracts tracking data directly and uses motmetrics/trackeval.
    """
    
    # Extract raw tracking data directly from VideoAnnotations
    gt_tracks = extract_tracks_direct(gt_annotation)
    pred_tracks = extract_tracks_direct(pred_annotation)
    
    logger.info(f"Extracted {len(gt_tracks)} GT frames, {len(pred_tracks)} pred frames")
    
    # Compute metrics
    basic_metrics = compute_basic_metrics_direct(gt_tracks, pred_tracks, iou_threshold)
    mot_metrics = compute_mot_metrics_direct(gt_tracks, pred_tracks, iou_threshold)
    
    # Combine results
    results = {
        'precision': basic_metrics['precision'],
        'recall': basic_metrics['recall'],
        'f1': basic_metrics['f1'],
        'avg_iou': basic_metrics['avg_iou'],
        
        'mota': mot_metrics['mota'],
        'motp': mot_metrics['motp'],
        'idf1': mot_metrics['idf1'],
        'id_switches': mot_metrics['id_switches'],
        'fragmentations': mot_metrics['fragmentations'],
        'num_misses': mot_metrics['num_misses'],
        'num_false_positives': mot_metrics['num_false_positives'],
        
        'hota': 0.0,  # TODO: implement if needed
        'deta': 0.0,
        'assa': 0.0,
        
        'true_positives': basic_metrics['tp'],
        'false_positives': basic_metrics['fp'],
        'false_negatives': basic_metrics['fn'],
        'total_gt_objects': basic_metrics['total_gt'],
        'total_pred_objects': basic_metrics['total_pred'],
        
        'iou_threshold': iou_threshold
    }
    
    logger.info(f"Direct evaluation - MOTA: {results['mota']:.3f}, Precision: {results['precision']:.3f}")
    return results


def extract_tracks_direct(annotation: VideoAnnotation) -> Dict[int, List[Dict]]:
    """Extract tracks directly without coordinate conversion."""
    tracks_by_frame = defaultdict(list)
    
    for frame in annotation.frames:
        frame_idx = frame.index
        
        for figure in frame.figures:
            # Get track ID
            if hasattr(figure.video_object, '_track_id'):
                track_id = figure.video_object._track_id
            else:
                track_id = figure.video_object.key().int
            
            # Get bbox directly from Rectangle geometry
            bbox = figure.geometry
            if not isinstance(bbox, sly.Rectangle):
                bbox = bbox.to_bbox()
            
            # Extract coordinates - avoid conversion errors
            left = float(bbox.left)
            top = float(bbox.top)
            right = float(bbox.right)
            bottom = float(bbox.bottom)
            
            track_obj = {
                'track_id': track_id,
                'bbox': [left, top, right, bottom],  # [x1, y1, x2, y2]
                'confidence': 1.0,
                'class_name': figure.video_object.obj_class.name
            }
            
            tracks_by_frame[frame_idx].append(track_obj)
    
    return dict(tracks_by_frame)


def compute_basic_metrics_direct(gt_tracks, pred_tracks, iou_threshold):
    """Compute basic metrics directly."""
    tp = fp = fn = 0
    total_iou = 0.0
    iou_count = 0
    
    all_frames = set(list(gt_tracks.keys()) + list(pred_tracks.keys()))
    
    for frame_idx in all_frames:
        gt_objects = gt_tracks.get(frame_idx, [])
        pred_objects = pred_tracks.get(frame_idx, [])
        
        if len(gt_objects) > 0 and len(pred_objects) > 0:
            gt_boxes = np.array([obj['bbox'] for obj in gt_objects])
            pred_boxes = np.array([obj['bbox'] for obj in pred_objects])
            
            iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)
            matches = find_matches(iou_matrix, iou_threshold)
            
            frame_tp = len(matches)
            frame_fp = len(pred_objects) - frame_tp
            frame_fn = len(gt_objects) - frame_tp
            
            tp += frame_tp
            fp += frame_fp
            fn += frame_fn
            
            for gt_idx, pred_idx in matches:
                total_iou += iou_matrix[gt_idx, pred_idx]
                iou_count += 1
        else:
            fp += len(pred_objects)
            fn += len(gt_objects)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    
    total_gt = sum(len(tracks) for tracks in gt_tracks.values())
    total_pred = sum(len(tracks) for tracks in pred_tracks.values())
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1, 'avg_iou': avg_iou,
        'tp': tp, 'fp': fp, 'fn': fn, 'total_gt': total_gt, 'total_pred': total_pred
    }


def compute_mot_metrics_direct(gt_tracks, pred_tracks, iou_threshold):
    """Compute MOT metrics directly using motmetrics."""
    if not MOTMETRICS_AVAILABLE:
        return {'mota': 0.0, 'motp': 0.0, 'idf1': 0.0, 'id_switches': 0, 'fragmentations': 0, 'num_misses': 0, 'num_false_positives': 0}
    
    try:
        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        
        for frame_idx in all_frames:
            gt_objects = gt_tracks.get(frame_idx, [])
            pred_objects = pred_tracks.get(frame_idx, [])
            
            gt_ids = [obj['track_id'] for obj in gt_objects]
            pred_ids = [obj['track_id'] for obj in pred_objects]
            
            if len(gt_objects) > 0 and len(pred_objects) > 0:
                gt_boxes = np.array([obj['bbox'] for obj in gt_objects])
                pred_boxes = np.array([obj['bbox'] for obj in pred_objects])
                
                iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)
                distance_matrix = 1.0 - iou_matrix
                distance_matrix[iou_matrix < iou_threshold] = np.inf
            else:
                distance_matrix = np.empty((len(gt_objects), len(pred_objects)))
                distance_matrix.fill(np.inf)
            
            acc.update(gt_ids, pred_ids, distance_matrix)
        
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches', 'num_fragmentations', 'num_misses', 'num_false_positives'], name='direct')
        
        def safe_extract(metric_name, default_value=0.0, as_int=False):
            if summary.empty or metric_name not in summary.columns:
                return int(default_value) if as_int else default_value
            value = summary[metric_name].iloc[0]
            if pd.isna(value):
                return int(default_value) if as_int else default_value
            return int(value) if as_int else float(value)
        
        return {
            'mota': safe_extract('mota', 0.0),
            'motp': safe_extract('motp', 0.0),
            'idf1': safe_extract('idf1', 0.0),
            'id_switches': safe_extract('num_switches', 0, as_int=True),
            'fragmentations': safe_extract('num_fragmentations', 0, as_int=True),
            'num_misses': safe_extract('num_misses', 0, as_int=True),
            'num_false_positives': safe_extract('num_false_positives', 0, as_int=True)
        }
        
    except Exception as e:
        logger.error(f"MOT metrics failed: {e}")
        return {'mota': 0.0, 'motp': 0.0, 'idf1': 0.0, 'id_switches': 0, 'fragmentations': 0, 'num_misses': 0, 'num_false_positives': 0}


def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculate IoU matrix between two sets of boxes."""
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = areas1[:, None] + areas2[None, :] - intersection
    
    return np.where(union > 0, intersection / union, 0)


def find_matches(iou_matrix: np.ndarray, threshold: float) -> List:
    """Find optimal matches using greedy assignment."""
    matches = []
    used_gt = set()
    used_pred = set()
    
    candidates = []
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            if iou_matrix[i, j] >= threshold:
                candidates.append((iou_matrix[i, j], i, j))
    
    candidates.sort(reverse=True)
    
    for iou_val, gt_idx, pred_idx in candidates:
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matches.append((gt_idx, pred_idx))
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
    
    return matches