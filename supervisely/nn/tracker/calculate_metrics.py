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
    
    Extracts tracks directly from VideoAnnotations and computes metrics
    using motmetrics and trackeval libraries.
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
        
        # Extract tracks directly from annotations
        gt_tracks = self._extract_tracks(gt_annotation)
        pred_tracks = self._extract_tracks(pred_annotation)
        
        if not gt_tracks:
            raise RuntimeError("No ground truth tracks found")
        if not pred_tracks:
            raise RuntimeError("No prediction tracks found")
        
        logger.info(f"Extracted {len(gt_tracks)} GT frames, {len(pred_tracks)} pred frames")
        
        # Compute all metrics
        basic_metrics = self._compute_basic_metrics(gt_tracks, pred_tracks)
        mot_metrics = self._compute_mot_metrics(gt_tracks, pred_tracks)
        hota_metrics = self._compute_hota_metrics(gt_tracks, pred_tracks)
        
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
    
    def _extract_tracks(self, annotation: VideoAnnotation) -> Dict[int, List[Dict]]:
        tracks_by_frame = defaultdict(list)
        
        for frame in annotation.frames:
            frame_idx = frame.index
            
            for figure in frame.figures:
                # Используем НАСТОЯЩИЙ track_id из VideoFigure
                if figure.track_id is not None:
                    track_id = int(figure.track_id)
                else:
                    # Fallback если track_id не установлен
                    track_id = figure.video_object.key().int
                
                # Get bbox directly from geometry
                bbox = figure.geometry
                if not isinstance(bbox, sly.Rectangle):
                    bbox = bbox.to_bbox()
                
                # Extract coordinates
                left = float(bbox.left)
                top = float(bbox.top)
                right = float(bbox.right)
                bottom = float(bbox.bottom)
                
                track_obj = {
                    'track_id': track_id,
                    'bbox': [left, top, right, bottom],
                    'confidence': 1.0,
                    'class_name': figure.video_object.obj_class.name
                }
                
                tracks_by_frame[frame_idx].append(track_obj)
        
        return dict(tracks_by_frame)
    
    def _compute_basic_metrics(self, gt_tracks, pred_tracks) -> Dict[str, float]:
        """Compute basic detection metrics."""
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
                
                iou_matrix = self._calculate_iou_matrix(gt_boxes, pred_boxes)
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
    
    def _compute_mot_metrics(self, gt_tracks, pred_tracks) -> Dict[str, Union[float, int]]:
        """Compute MOT metrics using motmetrics library."""
        if not MOTMETRICS_AVAILABLE:
            logger.warning("motmetrics not available - returning zero MOT metrics")
            return {
                'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                'id_switches': 0, 'fragmentations': 0,
                'num_misses': 0, 'num_false_positives': 0
            }
        
        try:
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
    
    def _compute_hota_metrics(self, gt_tracks, pred_tracks) -> Dict[str, float]:
        """Compute HOTA metrics using TrackEval library."""
        if not TRACKEVAL_AVAILABLE:
            logger.warning("trackeval not available - returning zero HOTA metrics")
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write MOT format files
                self._write_mot_files(gt_tracks, pred_tracks, temp_dir)
                
                # Run TrackEval
                return self._run_trackeval(temp_dir)
                
        except Exception as e:
            logger.error(f"Failed to compute HOTA metrics: {e}")
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
    
    def _write_mot_files(self, gt_tracks, pred_tracks, temp_dir: str):
        """Write tracking data in MOT format for TrackEval."""
        # Create directory structure
        gt_dir = Path(temp_dir) / 'gt' / 'MOT17-train' / 'seq' / 'gt'
        tracker_dir = Path(temp_dir) / 'trackers' / 'tracker' / 'MOT17-train'
        seqinfo_dir = Path(temp_dir) / 'gt' / 'MOT17-train' / 'seq'
        
        gt_dir.mkdir(parents=True, exist_ok=True)
        tracker_dir.mkdir(parents=True, exist_ok=True)
        
        # Write ground truth file
        with open(gt_dir / 'gt.txt', 'w') as f:
            for frame_idx in sorted(gt_tracks.keys()):
                for track in gt_tracks[frame_idx]:
                    x1, y1, x2, y2 = track['bbox']
                    width, height = x2 - x1, y2 - y1
                    
                    if width > 0 and height > 0:
                        line = f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{width:.1f},{height:.1f},1,1,1\n"
                        f.write(line)
        
        # Write tracker file
        with open(tracker_dir / 'seq.txt', 'w') as f:
            for frame_idx in sorted(pred_tracks.keys()):
                for track in pred_tracks[frame_idx]:
                    x1, y1, x2, y2 = track['bbox']
                    width, height = x2 - x1, y2 - y1
                    confidence = track.get('confidence', 1.0)
                    
                    if width > 0 and height > 0:
                        line = f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{width:.1f},{height:.1f},{confidence:.3f},1,1\n"
                        f.write(line)
        
        # Create seqinfo.ini
        max_frame = 0
        if gt_tracks:
            max_frame = max(max_frame, max(gt_tracks.keys()))
        if pred_tracks:
            max_frame = max(max_frame, max(pred_tracks.keys()))
        
        with open(seqinfo_dir / 'seqinfo.ini', 'w') as f:
            f.write("[Sequence]\n")
            f.write("name=seq\n")
            f.write(f"seqLength={max_frame + 1}\n")
            f.write("imWidth=1920\n")
            f.write("imHeight=1080\n")
            f.write("imExt=.jpg\n")
            f.write("imDir=img1\n")
            f.write("frameRate=30\n")
    
    def _run_trackeval(self, temp_dir: str) -> Dict[str, float]:
        """Run TrackEval to compute HOTA metrics."""
        try:
            eval_config = trackeval.Evaluator.get_default_eval_config()
            eval_config['DISPLAY_LESS_PROGRESS'] = True
            eval_config['USE_PARALLEL'] = False
            eval_config['PRINT_RESULTS'] = False
            
            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            dataset_config['GT_FOLDER'] = os.path.join(temp_dir, 'gt')
            dataset_config['TRACKERS_FOLDER'] = os.path.join(temp_dir, 'trackers')
            dataset_config['OUTPUT_FOLDER'] = os.path.join(temp_dir, 'output')
            dataset_config['TRACKERS_TO_EVAL'] = ['tracker']
            dataset_config['CLASSES_TO_EVAL'] = ['pedestrian']
            
            metrics_config = {'METRICS': ['HOTA']}
            
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            metrics_list = [trackeval.metrics.HOTA(metrics_config)]
            
            output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
            
            # Extract results
            if output_res:
                for dataset_name, dataset_res in output_res.items():
                    for tracker_name, tracker_res in dataset_res.items():
                        if 'COMBINED_SEQ' in tracker_res:
                            combined = tracker_res['COMBINED_SEQ']
                            if 'HOTA' in combined:
                                hota_data = combined['HOTA']
                                return {
                                    'hota': hota_data.get('HOTA', 0.0),
                                    'deta': hota_data.get('DetA', 0.0),
                                    'assa': hota_data.get('AssA', 0.0)
                                }
            
        except Exception as e:
            logger.warning(f"TrackEval failed: {e}")
        
        return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
    
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
    
    def _find_matches(self, iou_matrix: np.ndarray, threshold: float) -> List:
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
    