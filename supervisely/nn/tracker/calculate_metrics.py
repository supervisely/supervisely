import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import supervisely as sly
from supervisely import logger
from collections import defaultdict
import warnings

try:
    import motmetrics as mm
    import pandas as pd
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False
    warnings.warn("motmetrics not available. MOTA, HOTA metrics will be skipped.")


class TrackingEvaluator:
    """
    Unified evaluator for video tracking metrics from Supervisely annotations.
    Returns all metrics in a single JSON-compatible dictionary.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset internal state for new evaluation."""
        self._all_matches = []
        self._frame_data = {}
        self._class_names = set()
    
    def evaluate(
        self, 
        gt_annotation: sly.VideoAnnotation, 
        pred_annotation: sly.VideoAnnotation
    ) -> Dict[str, Union[float, int]]:
        """
        Evaluate all tracking metrics and return JSON result.
        
        Args:
            gt_annotation: Ground truth video annotation
            pred_annotation: Predicted video annotation
            
        Returns:
            Dictionary with all computed metrics in Supervisely format
            
        Raises:
            TypeError: If annotations are not VideoAnnotation objects
            ValueError: If annotations are empty or incompatible
        """
        # Validate inputs (same style as visualizer)
        if not isinstance(gt_annotation, sly.VideoAnnotation):
            raise TypeError(f"GT annotation must be VideoAnnotation, got {type(gt_annotation)}")
        
        if not isinstance(pred_annotation, sly.VideoAnnotation):
            raise TypeError(f"Prediction annotation must be VideoAnnotation, got {type(pred_annotation)}")
        
        self.reset()
        
        # Extract tracking data
        gt_tracks = self._extract_tracks(gt_annotation, is_gt=True)
        pred_tracks = self._extract_tracks(pred_annotation, is_gt=False)
        
        if not gt_tracks:
            logger.warning("No GT tracking data found in annotation")
        
        if not pred_tracks:
            logger.warning("No prediction tracking data found in annotation")
        
        # Calculate frame-by-frame matches
        self._calculate_matches(gt_tracks, pred_tracks)
        
        # Compute all metrics
        logger.info("Computing basic metrics...")
        basic_metrics = self._compute_basic_metrics()
        
        logger.info("Computing tracking metrics...")
        tracking_metrics = self._compute_tracking_metrics(gt_tracks, pred_tracks)
        
        # Combine results in Supervisely format
        result = {
            # Basic detection metrics
            "precision": basic_metrics["precision"],
            "recall": basic_metrics["recall"],
            "f1": basic_metrics["f1"],
            "iou": basic_metrics["avg_iou"],
            
            # Tracking-specific metrics
            "mota": tracking_metrics.get("mota", 0.0),
            "motp": tracking_metrics.get("motp", 0.0),
            "idf1": tracking_metrics.get("idf1", 0.0),
            "num_switches": tracking_metrics.get("num_switches", 0),
            "num_misses": tracking_metrics.get("num_misses", 0),
            "num_false_positives": tracking_metrics.get("num_false_positives", 0),
            
            # Counts for analysis
            "true_positives": basic_metrics["tp"],
            "false_positives": basic_metrics["fp"], 
            "false_negatives": basic_metrics["fn"],
            "total_gt_objects": basic_metrics["total_gt"],
            "total_pred_objects": basic_metrics["total_pred"],
            
            # Configuration
            "iou_threshold": self.iou_threshold,
            "frames_evaluated": len(self._frame_data),
        }
        
        logger.info(f"Evaluation complete. MOTA: {result['mota']:.3f}, "
                   f"Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
        
        return result
    
    def _extract_tracks(self, annotation: sly.VideoAnnotation, is_gt: bool) -> Dict[int, List]:
        """
        Extract track data from video annotation.
        Uses the same approach as VideoTrackingVisualizer for consistency.
        
        Returns:
            Dictionary mapping frame_idx -> list of track objects
        """
        tracks_by_frame = defaultdict(list)
        
        # Map object keys to track info (same as visualizer)
        objects = {}
        for i, obj in enumerate(annotation.objects):
            objects[obj.key] = (i, obj.obj_class.name)
        
        # Extract tracks from frames (same logic as visualizer)
        for frame in annotation.frames:
            frame_idx = frame.index
            for figure in frame.figures:
                # Only process rectangles (same as visualizer)
                if figure.geometry.geometry_name() != 'rectangle':
                    continue
                    
                object_key = figure.parent_object.key
                if object_key not in objects:
                    continue
                    
                track_id, class_name = objects[object_key]
                self._class_names.add(class_name)
                
                # Extract bbox coordinates (same as visualizer)
                rect = figure.geometry
                bbox = [rect.left, rect.top, rect.right, rect.bottom]
                
                # Create track object with additional metadata for evaluation
                track_obj = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'class': class_name,
                    'is_gt': is_gt,
                    'confidence': 1.0  # Default confidence for GT
                }
                
                # Extract confidence from tags if available
                if not is_gt:
                    # Check figure tags first
                    confidence = self._extract_confidence_from_tags(figure.tags)
                    if confidence is None:
                        # Fall back to object tags
                        confidence = self._extract_confidence_from_tags(figure.parent_object.tags)
                    if confidence is not None:
                        track_obj['confidence'] = confidence
                
                tracks_by_frame[frame_idx].append(track_obj)
        
        logger.info(f"Extracted tracks from {len(tracks_by_frame)} frames ({'GT' if is_gt else 'Pred'})")
        return dict(tracks_by_frame)
    
    def _extract_confidence_from_tags(self, tags) -> Optional[float]:
        """Extract confidence value from tags collection."""
        for tag in tags:
            if tag.meta.name in ['confidence', 'conf', 'score']:
                return float(tag.value)
        return None
    
    def _calculate_matches(self, gt_tracks: Dict, pred_tracks: Dict):
        """Calculate matches between GT and predictions for each frame."""
        all_frames = set(list(gt_tracks.keys()) + list(pred_tracks.keys()))
        
        logger.info(f"Calculating matches for {len(all_frames)} frames")
        
        for frame_idx in all_frames:
            gt_frame = gt_tracks.get(frame_idx, [])
            pred_frame = pred_tracks.get(frame_idx, [])
            
            frame_matches = self._match_frame(gt_frame, pred_frame)
            self._frame_data[frame_idx] = {
                'gt': gt_frame,
                'pred': pred_frame,
                'matches': frame_matches
            }
            
            self._all_matches.extend(frame_matches)
        
        logger.info(f"Found {len(self._all_matches)} total matches")
    
    def _match_frame(self, gt_objects: List, pred_objects: List) -> List[Dict]:
        """Match predictions to ground truth for a single frame."""
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        # Calculate IoU matrix
        for i, pred_obj in enumerate(pred_objects):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt_obj in enumerate(gt_objects):
                if j in matched_gt or pred_obj['class'] != gt_obj['class']:
                    continue
                
                iou = self._calculate_iou(pred_obj['bbox'], gt_obj['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                # True Positive
                matches.append({
                    'type': 'TP',
                    'pred_idx': i,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'pred_track_id': pred_objects[i]['track_id'],
                    'gt_track_id': gt_objects[best_gt_idx]['track_id'],
                    'class': pred_objects[i]['class'],
                    'confidence': pred_objects[i]['confidence']
                })
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                # False Positive
                matches.append({
                    'type': 'FP',
                    'pred_idx': i,
                    'pred_track_id': pred_objects[i]['track_id'],
                    'class': pred_objects[i]['class'],
                    'confidence': pred_objects[i]['confidence']
                })
        
        # False Negatives
        for j, gt_obj in enumerate(gt_objects):
            if j not in matched_gt:
                matches.append({
                    'type': 'FN',
                    'gt_idx': j,
                    'gt_track_id': gt_obj['track_id'],
                    'class': gt_obj['class']
                })
        
        return matches
    
    def _compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic detection metrics."""
        tp_matches = [m for m in self._all_matches if m['type'] == 'TP']
        fp_matches = [m for m in self._all_matches if m['type'] == 'FP']
        fn_matches = [m for m in self._all_matches if m['type'] == 'FN']
        
        tp = len(tp_matches)
        fp = len(fp_matches)
        fn = len(fn_matches)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Average IoU of true positives
        avg_iou = np.mean([m['iou'] for m in tp_matches]) if tp_matches else 0.0
        
        # Count total objects
        total_gt = len([m for m in self._all_matches if m['type'] in ['TP', 'FN']])
        total_pred = len([m for m in self._all_matches if m['type'] in ['TP', 'FP']])
        
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
    
    def _compute_tracking_metrics(self, gt_tracks: Dict, pred_tracks: Dict) -> Dict[str, float]:
        """Compute advanced tracking metrics using motmetrics."""
        if not MOTMETRICS_AVAILABLE:
            logger.warning("motmetrics not available, skipping advanced tracking metrics")
            return {}
        
        try:
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Process each frame
            frame_count = 0
            for frame_idx in sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys()))):
                gt_frame = gt_tracks.get(frame_idx, [])
                pred_frame = pred_tracks.get(frame_idx, [])
                
                if not gt_frame and not pred_frame:
                    continue
                
                # Extract track IDs and bounding boxes
                gt_ids = [obj['track_id'] for obj in gt_frame]
                pred_ids = [obj['track_id'] for obj in pred_frame]
                
                gt_boxes = np.array([obj['bbox'] for obj in gt_frame]) if gt_frame else np.empty((0, 4))
                pred_boxes = np.array([obj['bbox'] for obj in pred_frame]) if pred_frame else np.empty((0, 4))
                
                # Calculate distance matrix (1 - IoU)
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    distances = self._calculate_distance_matrix(gt_boxes, pred_boxes)
                else:
                    distances = np.empty((len(gt_boxes), len(pred_boxes)))
                
                # Update accumulator
                acc.update(gt_ids, pred_ids, distances)
                frame_count += 1
            
            logger.info(f"Processed {frame_count} frames for tracking metrics")
            
            # Calculate metrics
            mh = mm.metrics.create()
            summary = mh.compute(
                acc, 
                metrics=['mota', 'motp', 'idf1', 'num_switches', 'num_misses', 'num_false_positives'], 
                name='tracking_eval'
            )
            
            if summary.empty:
                logger.warning("No tracking metrics computed - empty summary")
                return {}
            
            return {
                'mota': float(summary['mota'].iloc[0]) if not pd.isna(summary['mota'].iloc[0]) else 0.0,
                'motp': float(summary['motp'].iloc[0]) if not pd.isna(summary['motp'].iloc[0]) else 0.0,
                'idf1': float(summary['idf1'].iloc[0]) if not pd.isna(summary['idf1'].iloc[0]) else 0.0,
                'num_switches': int(summary['num_switches'].iloc[0]),
                'num_misses': int(summary['num_misses'].iloc[0]),
                'num_false_positives': int(summary['num_false_positives'].iloc[0])
            }
            
        except Exception as e:
            logger.warning(f"Failed to compute tracking metrics: {e}")
            return {}
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance_matrix(self, gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """Calculate distance matrix (1 - IoU) for motmetrics."""
        distances = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou = self._calculate_iou(gt_box.tolist(), pred_box.tolist())
                distances[i, j] = 1.0 - iou
        
        return distances


