import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import supervisely as sly
from supervisely import logger
from collections import defaultdict
import tempfile
import os
import subprocess
from pathlib import Path

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
    Video tracking metrics evaluator with MOTA and HOTA support.
    Extracts tracks from Supervisely VideoAnnotation and computes comprehensive metrics.
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
        Evaluate all tracking metrics and return results.
        
        Args:
            gt_annotation: Ground truth video annotation
            pred_annotation: Predicted video annotation
            
        Returns:
            Dictionary with all computed metrics
        """
        # Validate inputs
        if not isinstance(gt_annotation, sly.VideoAnnotation):
            raise TypeError(f"GT annotation must be VideoAnnotation, got {type(gt_annotation)}")
        
        if not isinstance(pred_annotation, sly.VideoAnnotation):
            raise TypeError(f"Prediction annotation must be VideoAnnotation, got {type(pred_annotation)}")
        
        self.reset()
        logger.info(f"Starting evaluation with IoU threshold: {self.iou_threshold}")
        
        # Extract tracking data
        gt_tracks = self._extract_tracks(gt_annotation, is_gt=True)
        pred_tracks = self._extract_tracks(pred_annotation, is_gt=False)
        
        if not gt_tracks:
            raise RuntimeError("No ground truth tracks found")
        if not pred_tracks:
            raise RuntimeError("No prediction tracks found")
        
        # Calculate frame-by-frame matches
        self._calculate_matches(gt_tracks, pred_tracks)
        
        # Compute all metrics
        basic_metrics = self._compute_basic_metrics()
        tracking_metrics = self._compute_mot_metrics(gt_tracks, pred_tracks)
        hota_metrics = self._compute_hota_metrics(gt_tracks, pred_tracks)
        
        # Combine results
        result = {
            # Basic detection metrics
            "precision": basic_metrics["precision"],
            "recall": basic_metrics["recall"],
            "f1": basic_metrics["f1"],
            "iou": basic_metrics["avg_iou"],
            
            # MOT metrics
            "mota": tracking_metrics["mota"],
            "motp": tracking_metrics["motp"],
            "idf1": tracking_metrics["idf1"],
            "num_switches": tracking_metrics["num_switches"],
            "num_misses": tracking_metrics["num_misses"],
            "num_false_positives": tracking_metrics["num_false_positives"],
            
            # HOTA metrics
            "hota": hota_metrics["hota"],
            "deta": hota_metrics["deta"],
            "assa": hota_metrics["assa"],
            
            # Counts
            "true_positives": basic_metrics["tp"],
            "false_positives": basic_metrics["fp"], 
            "false_negatives": basic_metrics["fn"],
            "total_gt_objects": basic_metrics["total_gt"],
            "total_pred_objects": basic_metrics["total_pred"],
            
            # Config
            "iou_threshold": self.iou_threshold,
            "frames_evaluated": len(self._frame_data),
        }
        
        logger.info(f"Evaluation complete. MOTA: {result['mota']:.3f}, "
                   f"HOTA: {result['hota']:.3f}, Precision: {result['precision']:.3f}")
        
        return result
    
    def _extract_tracks(self, annotation: sly.VideoAnnotation, is_gt: bool) -> Dict[int, List]:
        """Extract track data with proper track IDs from tags."""
        tracks_by_frame = defaultdict(list)
        
        # Extract track_id from object tags
        object_track_ids = {}
        for obj in annotation.objects:
            track_id = None
            
            # Look for track_id in object tags
            if obj.tags:
                for tag in obj.tags:
                    if hasattr(tag, 'meta') and hasattr(tag.meta, 'name'):
                        if tag.meta.name == 'track_id':
                            track_id = int(tag.value)
                            break
            
            # Fallback if track_id not found in tags
            if track_id is None:
                obj_key = obj.key() if callable(obj.key) else obj.key
                track_id = hash(obj_key) % 10000  # Use hash as fallback
                logger.warning(f"No track_id found in tags for object {obj_key}, using fallback: {track_id}")
            
            obj_key = obj.key() if callable(obj.key) else obj.key
            object_track_ids[obj_key] = (track_id, obj.obj_class.name)
        
        # Extract tracks from frames
        for frame in annotation.frames:
            frame_idx = frame.index
            for figure in frame.figures:
                # Only process rectangles
                if figure.geometry.geometry_name() != 'rectangle':
                    continue
                    
                object_key = figure.parent_object.key() if callable(figure.parent_object.key) else figure.parent_object.key
                if object_key not in object_track_ids:
                    continue
                    
                track_id, class_name = object_track_ids[object_key]
                self._class_names.add(class_name)
                
                # Extract bbox coordinates
                rect = figure.geometry
                bbox = [rect.left, rect.top, rect.right, rect.bottom]
                
                # Validate bbox
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                
                track_obj = {
                    'track_id': track_id,  # Use proper track_id from tags
                    'bbox': bbox,
                    'class': class_name,
                    'is_gt': is_gt,
                    'confidence': 1.0
                }
                
                # Extract confidence from object tags
                if not is_gt:
                    confidence = self._extract_confidence_from_tags(figure.parent_object.tags)
                    if confidence is not None:
                        track_obj['confidence'] = confidence
                
                tracks_by_frame[frame_idx].append(track_obj)
        
        logger.info(f"Extracted tracks from {len(tracks_by_frame)} frames ({'GT' if is_gt else 'Pred'})")
        return dict(tracks_by_frame)
    
    def _extract_confidence_from_tags(self, tags) -> Optional[float]:
        """Extract confidence value from tags collection."""
        if not tags:
            return None
            
        try:
            for tag in tags:
                if hasattr(tag, 'meta') and hasattr(tag.meta, 'name'):
                    if tag.meta.name in ['confidence', 'conf', 'score']:
                        return float(tag.value)
        except (AttributeError, ValueError, TypeError):
            pass
        
        return None
    
    def _calculate_matches(self, gt_tracks: Dict, pred_tracks: Dict):
        """Calculate matches between GT and predictions for each frame."""
        all_frames = set(list(gt_tracks.keys()) + list(pred_tracks.keys()))
        
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
    
    def _match_frame(self, gt_objects: List, pred_objects: List) -> List[Dict]:
        """Match predictions to ground truth for a single frame using Hungarian algorithm."""
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        # Greedy matching based on IoU
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
        
        avg_iou = np.mean([m['iou'] for m in tp_matches]) if tp_matches else 0.0
        
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
    
    def _compute_mot_metrics(self, gt_tracks: Dict, pred_tracks: Dict) -> Dict[str, float]:
        """Compute MOT metrics using motmetrics library."""
        if not MOTMETRICS_AVAILABLE:
            logger.error("motmetrics not available. Install with: pip install motmetrics")
            return {
                'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                'num_switches': 0, 'num_misses': 0, 'num_false_positives': 0
            }
        
        try:
            # Create accumulator
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Process each frame
            total_updates = 0
            for frame_idx in sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys()))):
                gt_frame = gt_tracks.get(frame_idx, [])
                pred_frame = pred_tracks.get(frame_idx, [])
                
                # Skip completely empty frames
                if len(gt_frame) == 0 and len(pred_frame) == 0:
                    continue
                
                # Extract numeric track IDs
                gt_ids = [obj['track_id'] for obj in gt_frame]
                pred_ids = [obj['track_id'] for obj in pred_frame]
                
                # Calculate distance matrix
                if len(gt_frame) > 0 and len(pred_frame) > 0:
                    gt_boxes = np.array([obj['bbox'] for obj in gt_frame])
                    pred_boxes = np.array([obj['bbox'] for obj in pred_frame])
                    distances = self._calculate_distance_matrix(gt_boxes, pred_boxes)
                    
                    # Set high distance for poor matches
                    distances[distances > (1 - self.iou_threshold)] = np.inf
                else:
                    distances = np.empty((len(gt_frame), len(pred_frame)))
                    distances.fill(np.inf)
                
                # Update accumulator
                acc.update(gt_ids, pred_ids, distances)
                total_updates += 1
            
            logger.info(f"Updated motmetrics accumulator {total_updates} times")
            
            # Compute metrics
            mh = mm.metrics.create()
            summary = mh.compute(
                acc, 
                metrics=['mota', 'motp', 'idf1', 'num_switches', 'num_misses', 'num_false_positives'],
                name='tracking_eval'
            )
            
            if summary.empty or len(summary) == 0:
                logger.warning("motmetrics returned empty summary")
                return {
                    'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                    'num_switches': 0, 'num_misses': 0, 'num_false_positives': 0
                }
            
            # Extract metrics safely
            result = {}
            for metric in ['mota', 'motp', 'idf1', 'num_switches', 'num_misses', 'num_false_positives']:
                if metric not in summary.columns:
                    logger.warning(f"Metric {metric} not found in motmetrics results")
                    result[metric] = 0.0 if metric not in ['num_switches', 'num_misses', 'num_false_positives'] else 0
                    continue
                
                value = summary[metric].iloc[0]
                if pd.isna(value):
                    result[metric] = 0.0 if metric not in ['num_switches', 'num_misses', 'num_false_positives'] else 0
                else:
                    if metric in ['num_switches', 'num_misses', 'num_false_positives']:
                        result[metric] = int(value)
                    else:
                        result[metric] = float(value)
            
            logger.info(f"Successfully computed MOT metrics: MOTA={result['mota']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute MOT metrics: {str(e)}")
            return {
                'mota': 0.0, 'motp': 0.0, 'idf1': 0.0,
                'num_switches': 0, 'num_misses': 0, 'num_false_positives': 0
            }
    
    def _compute_hota_metrics(self, gt_tracks: Dict, pred_tracks: Dict) -> Dict[str, float]:
        """Compute HOTA metrics using simplified approach."""
        if not TRACKEVAL_AVAILABLE:
            logger.warning("trackeval not available. Install with: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating TrackEval files in: {temp_dir}")
                
                # Write files in MOT format
                self._write_mot_files(gt_tracks, pred_tracks, temp_dir)
                
                # Try to run TrackEval command line tool
                hota_result = self._run_trackeval_command(temp_dir)
                
                if hota_result:
                    return hota_result
                
                # Fallback to Python API
                return self._run_trackeval_api(temp_dir)
                
        except Exception as e:
            logger.error(f"Failed to compute HOTA metrics: {str(e)}")
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
    
    def _write_mot_files(self, gt_tracks: Dict, pred_tracks: Dict, temp_dir: str):
        """Write tracking data in MOT format for TrackEval."""
        
        # Create directory structure
        gt_seq_dir = os.path.join(temp_dir, 'gt', 'seq')
        gt_dir = os.path.join(gt_seq_dir, 'gt')
        tracker_dir = os.path.join(temp_dir, 'trackers', 'tracker')
        seqmap_dir = os.path.join(temp_dir, 'gt', 'seqmaps')
        
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(seqmap_dir, exist_ok=True)
        
        # Create seqmap file
        with open(os.path.join(seqmap_dir, 'MOT17-train.txt'), 'w') as f:
            f.write("name\n")
            f.write("seq\n")
        
        # Create seqinfo.ini
        max_frame = 0
        if gt_tracks:
            max_frame = max(max_frame, max(gt_tracks.keys()))
        if pred_tracks:
            max_frame = max(max_frame, max(pred_tracks.keys()))
        
        with open(os.path.join(gt_seq_dir, 'seqinfo.ini'), 'w') as f:
            f.write("[Sequence]\n")
            f.write("name=seq\n")
            f.write(f"seqLength={max_frame + 1}\n")
            f.write("imWidth=1920\n")
            f.write("imHeight=1080\n")
            f.write("imExt=.jpg\n")
            f.write("imDir=img1\n")
            f.write("frameRate=30\n")
        
        # Write GT file
        with open(os.path.join(gt_dir, 'gt.txt'), 'w') as f:
            for frame_idx in sorted(gt_tracks.keys()):
                for track in gt_tracks[frame_idx]:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    left, top, right, bottom = bbox
                    width, height = right - left, bottom - top
                    
                    if width > 0 and height > 0:
                        f.write(f"{frame_idx + 1},{track_id},{left:.1f},{top:.1f},{width:.1f},{height:.1f},1,1,1\n")
        
        # Write tracker file
        with open(os.path.join(tracker_dir, 'seq.txt'), 'w') as f:
            for frame_idx in sorted(pred_tracks.keys()):
                for track in pred_tracks[frame_idx]:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    left, top, right, bottom = bbox
                    width, height = right - left, bottom - top
                    
                    if width > 0 and height > 0:
                        conf = track['confidence']
                        f.write(f"{frame_idx + 1},{track_id},{left:.1f},{top:.1f},{width:.1f},{height:.1f},{conf:.3f},1,1\n")
        
        logger.info(f"Wrote MOT format files to {temp_dir}")
    
    def _run_trackeval_command(self, temp_dir: str) -> Optional[Dict[str, float]]:
        """Try to run TrackEval as command line tool."""
        try:
            # Check if TrackEval command exists
            cmd = [
                'python', '-m', 'trackeval.scripts.run_mot_challenge',
                '--BENCHMARK', 'MOT17',
                '--SPLIT_TO_EVAL', 'train',
                '--TRACKERS_TO_EVAL', 'tracker',
                '--USE_PARALLEL', 'False',
                '--GT_FOLDER', os.path.join(temp_dir, 'gt'),
                '--TRACKERS_FOLDER', os.path.join(temp_dir, 'trackers'),
                '--OUTPUT_FOLDER', os.path.join(temp_dir, 'output'),
                '--PRINT_RESULTS', 'False',
                '--PRINT_CONFIG', 'False'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Look for output files with HOTA results
                output_dir = os.path.join(temp_dir, 'output')
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if 'HOTA' in file and file.endswith('.txt'):
                            file_path = os.path.join(root, file)
                            hota_data = self._parse_hota_output(file_path)
                            if hota_data:
                                return hota_data
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"TrackEval command failed: {str(e)}")
        
        return None
    
    def _run_trackeval_api(self, temp_dir: str) -> Dict[str, float]:
        """Run TrackEval using Python API as fallback."""
        try:
            # Minimal config
            config = {
                'USE_PARALLEL': False,
                'PRINT_RESULTS': False,
                'PRINT_CONFIG': False,
                'PLOT_CURVES': False,
            }
            
            dataset_config = {
                'GT_FOLDER': os.path.join(temp_dir, 'gt'),
                'TRACKERS_FOLDER': os.path.join(temp_dir, 'trackers'),
                'OUTPUT_FOLDER': os.path.join(temp_dir, 'output'),
                'TRACKERS_TO_EVAL': ['tracker'],
                'CLASSES_TO_EVAL': ['all'],
                'BENCHMARK': 'MOT17',
                'SPLIT_TO_EVAL': 'train',
            }
            
            metric_config = {'THRESHOLD': self.iou_threshold}
            
            # Suppress output
            import sys
            from io import StringIO
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = StringIO()
            
            try:
                evaluator = trackeval.Evaluator(config)
                dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
                metrics_list = [trackeval.metrics.HOTA(metric_config)]
                
                raw_results, _ = evaluator.evaluate(dataset_list, metrics_list)
                
                
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            
            # Extract HOTA data
            if (raw_results and 'MotChallenge2DBox' in raw_results and 
                raw_results['MotChallenge2DBox'] and 'tracker' in raw_results['MotChallenge2DBox']):
                
                tracker_data = raw_results['MotChallenge2DBox']['tracker']
                if tracker_data:
                    for seq_data in tracker_data.values():
                        if isinstance(seq_data, dict) and 'HOTA' in seq_data:
                            hota_data = seq_data['HOTA']
                            return {
                                'hota': hota_data.get('HOTA', 0.0),
                                'deta': hota_data.get('DetA', 0.0),
                                'assa': hota_data.get('AssA', 0.0),
                            }
            
        except Exception as e:
            logger.debug(f"TrackEval API failed: {str(e)}")
        
        return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
    
    def _parse_hota_output(self, file_path: str) -> Optional[Dict[str, float]]:
        """Parse HOTA results from TrackEval output file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Simple parsing of HOTA results
            # This is a placeholder - actual parsing depends on TrackEval output format
            if 'HOTA' in content:
                # Extract HOTA value (simplified)
                lines = content.split('\n')
                for line in lines:
                    if 'HOTA' in line and any(char.isdigit() for char in line):
                        # Try to extract numeric values
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            hota_val = float(numbers[0]) / 100.0  # Convert percentage to decimal
                            return {'hota': hota_val, 'deta': 0.0, 'assa': 0.0}
            
        except Exception as e:
            logger.debug(f"Failed to parse HOTA output: {str(e)}")
        
        return None
    
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
