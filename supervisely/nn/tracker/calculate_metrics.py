import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Union
from collections import defaultdict
import supervisely as sly
from supervisely import logger
from supervisely.video_annotation.video_annotation import VideoAnnotation

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
        self.img_height, self.img_width = gt_annotation.img_size
        
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
        # hota_metrics = self._compute_hota_metrics(gt_tracks, pred_tracks)
        
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
            
            # # HOTA metrics
            # 'hota': hota_metrics['hota'],
            # 'deta': hota_metrics['deta'],
            # 'assa': hota_metrics['assa'],
            
            # Count metrics
            'true_positives': basic_metrics['tp'],
            'false_positives': basic_metrics['fp'],
            'false_negatives': basic_metrics['fn'],
            'total_gt_objects': basic_metrics['total_gt'],
            'total_pred_objects': basic_metrics['total_pred'],
            
            # Config
            'iou_threshold': self.iou_threshold
        }
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
        try:
            import motmetrics as mm
        except ImportError:
            logger.error(
                "motmetrics not available. Install with: pip install motmetrics"
            )
            raise
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
    
    # def _compute_hota_metrics(self, gt_tracks, pred_tracks) -> Dict[str, float]:
    #     """Compute HOTA metrics using TrackEval library."""
    #     try:
    #         import trackeval
    #     except ImportError:
    #         logger.error(
    #             "trackeval not available. Install with: ip install git+https://github.com/JonathonLuiten/TrackEval.git"
    #         )
    #         raise

    #     try:
    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             # Write MOT format files
    #             self._write_mot_files(gt_tracks, pred_tracks, temp_dir)
                
    #             # Run TrackEval
    #             return self._run_trackeval(temp_dir)
                
    #     except Exception as e:
    #         logger.error(f"Failed to compute HOTA metrics: {e}")
    #         return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
        
    # def _write_mot_files(self, gt_tracks, pred_tracks, temp_dir: str):
    #     """Write tracking data in MOT format for TrackEval with both canonical and benchmark-style layouts.

    #     This function creates:
    #     - <temp_dir>/gt/seq/gt/gt.txt and <temp_dir>/trackers/tracker/data/seq.txt and seqmap in gt/
    #     - <temp_dir>/MOT17/train/gt/seq/gt.txt and <temp_dir>/MOT17/train/trackers/tracker/data/seq.txt and seqmap in MOT17/train/gt/
    #     This guarantees compatibility with different TrackEval versions that expect benchmark/split folders.
    #     """
    #     from pathlib import Path

    #     seq_name = 'seq'  # consistent sequence name used everywhere
    #     # Primary canonical locations (existing behavior)
    #     canonical_gt_seq = Path(temp_dir) / 'gt' / seq_name
    #     canonical_gt_data = canonical_gt_seq / 'gt'
    #     canonical_tracker_data = Path(temp_dir) / 'trackers' / 'tracker' / 'data'
    #     canonical_seqmaps = Path(temp_dir) / 'gt' / 'seqmaps'

    #     # Benchmark-style locations (what TrackEval's MotChallenge2DBox often expects)
    #     bench_root = Path(temp_dir) / 'MOT17' / 'train'
    #     bench_gt_seq = bench_root / 'gt' / seq_name
    #     bench_gt_data = bench_gt_seq / 'gt'
    #     bench_tracker_data = bench_root / 'trackers' / 'tracker' / 'data'
    #     bench_seqmaps = bench_root / 'gt' / 'seqmaps'

    #     # Create directories (both)
    #     for d in (canonical_gt_data, canonical_tracker_data, canonical_seqmaps,
    #             bench_gt_data, bench_tracker_data, bench_seqmaps):
    #         d.mkdir(parents=True, exist_ok=True)

    #     # Helper to write GT file to a target path
    #     def write_gt_file(target_gt_file: Path):
    #         with open(target_gt_file, 'w') as f:
    #             for frame_idx in sorted(gt_tracks.keys()):
    #                 for track in gt_tracks[frame_idx]:
    #                     x1, y1, x2, y2 = track['bbox']
    #                     w, h = x2 - x1, y2 - y1
    #                     if w > 0 and h > 0:
    #                         line = f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,1,1\n"
    #                         f.write(line)

    #     # Helper to write tracker file
    #     def write_tracker_file(target_tracker_file: Path):
    #         with open(target_tracker_file, 'w') as f:
    #             for frame_idx in sorted(pred_tracks.keys()):
    #                 for track in pred_tracks[frame_idx]:
    #                     x1, y1, x2, y2 = track['bbox']
    #                     w, h = x2 - x1, y2 - y1
    #                     conf = track.get('confidence', 1.0)
    #                     if w > 0 and h > 0:
    #                         line = f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{conf:.3f},1,1\n"
    #                         f.write(line)

    #     # Write canonical GT and tracker
    #     canonical_gt_file = canonical_gt_data / 'gt.txt'
    #     write_gt_file(canonical_gt_file)
    #     canonical_tracker_file = canonical_tracker_data / f'{seq_name}.txt'
    #     write_tracker_file(canonical_tracker_file)

    #     # Write benchmark-style GT and tracker (duplicate)
    #     bench_gt_file = bench_gt_data / 'gt.txt'
    #     write_gt_file(bench_gt_file)
    #     bench_tracker_file = bench_tracker_data / f'{seq_name}.txt'
    #     write_tracker_file(bench_tracker_file)

    #     # Compute seqLength for seqinfo.ini (max frame index present)
    #     max_frame = 0
    #     if gt_tracks:
    #         max_frame = max(max_frame, max(gt_tracks.keys()))
    #     if pred_tracks:
    #         max_frame = max(max_frame, max(pred_tracks.keys()))

    #     # Write seqinfo.ini in both places
    #     seqinfo_text = (
    #         "[Sequence]\n"
    #         f"name={seq_name}\n"
    #         f"seqLength={max_frame + 1}\n"
    #         f"imWidth={self.img_width}\n"
    #         f"imHeight={self.img_height}\n"
    #         "imExt=.jpg\n"
    #         "imDir=img1\n"
    #         "frameRate=30\n"
    #     )
    #     with open(canonical_gt_seq / 'seqinfo.ini', 'w') as f:
    #         f.write(seqinfo_text)
    #     with open(bench_gt_seq / 'seqinfo.ini', 'w') as f:
    #         f.write(seqinfo_text)

    #     # Write seqmap files (one line per sequence name) in both canonical and benchmark locations
    #     seqmap_lines = [f"{seq_name}\n"]
    #     with open(Path(temp_dir) / 'gt' / 'seqmap.txt', 'w') as f:
    #         f.writelines(seqmap_lines)
    #     with open(canonical_seqmaps / 'seqmap.txt', 'w') as f:
    #         f.writelines(seqmap_lines)
    #     with open(bench_gt_seq / 'seqmap.txt', 'w') as f:
    #         f.writelines(seqmap_lines)
    #     with open(bench_seqmaps / 'seqmap.txt', 'w') as f:
    #         f.writelines(seqmap_lines)

    #     # Debug logging
    #     logger.info(f"Created canonical GT: {canonical_gt_file}")
    #     logger.info(f"Created canonical tracker: {canonical_tracker_file}")
    #     logger.info(f"Created benchmark GT: {bench_gt_file}")
    #     logger.info(f"Created benchmark tracker: {bench_tracker_file}")
    #     logger.info(f"Wrote seqmap to: {Path(temp_dir)/'gt'/'seqmap.txt'} and {bench_gt_seq/'seqmap.txt'}")

    # def _run_trackeval(self, temp_dir: str) -> Dict[str, float]:
    #     """Run TrackEval to compute HOTA metrics. Robust: create MOT structure, chdir, try DO_PREPROC False->True."""
    #     try:
    #         # prepare basic eval config
    #         eval_config = trackeval.Evaluator.get_default_eval_config()
    #         eval_config['DISPLAY_LESS_PROGRESS'] = True
    #         eval_config['USE_PARALLEL'] = False
    #         eval_config['PRINT_RESULTS'] = False
    #         eval_config['PRINT_CONFIG'] = False

    #         # make sure required folders/files exist in canonical places
    #         abs_gt = os.path.join(temp_dir, 'gt')
    #         abs_trackers = os.path.join(temp_dir, 'trackers')
    #         abs_output = os.path.join(temp_dir, 'output')
    #         seq_name = 'seq'  # consistent sequence name

    #         # create canonical MOT layout
    #         gt_seq_dir = Path(abs_gt) / seq_name
    #         gt_data_dir = gt_seq_dir / 'gt'
    #         tracker_data_dir = Path(abs_trackers) / 'tracker' / 'data'
    #         seqmaps_dir = Path(abs_gt) / 'seqmaps'

    #         gt_data_dir.mkdir(parents=True, exist_ok=True)
    #         tracker_data_dir.mkdir(parents=True, exist_ok=True)
    #         seqmaps_dir.mkdir(parents=True, exist_ok=True)
    #         Path(abs_output).mkdir(parents=True, exist_ok=True)

    #         # write gt file (if not already)
    #         gt_file = gt_data_dir / 'gt.txt'
    #         if not gt_file.exists():
    #             with open(gt_file, 'w') as f:
    #                 # expect gt_tracks created earlier by caller; if not, leave empty
    #                 if hasattr(self, '_last_gt_tracks') and self._last_gt_tracks:
    #                     for frame_idx in sorted(self._last_gt_tracks.keys()):
    #                         for track in self._last_gt_tracks[frame_idx]:
    #                             x1, y1, x2, y2 = track['bbox']
    #                             w, h = x2 - x1, y2 - y1
    #                             if w > 0 and h > 0:
    #                                 f.write(f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,1,1\n")
    #                 else:
    
    #                     pass

    #         # write tracker file
    #         tracker_file = tracker_data_dir / f"{seq_name}.txt"
    #         if not tracker_file.exists():
    #             with open(tracker_file, 'w') as f:
    #                 if hasattr(self, '_last_pred_tracks') and self._last_pred_tracks:
    #                     for frame_idx in sorted(self._last_pred_tracks.keys()):
    #                         for track in self._last_pred_tracks[frame_idx]:
    #                             x1, y1, x2, y2 = track['bbox']
    #                             w, h = x2 - x1, y2 - y1
    #                             conf = track.get('confidence', 1.0)
    #                             if w > 0 and h > 0:
    #                                 f.write(f"{frame_idx + 1},{track['track_id']},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{conf:.3f},1,1\n")

    #         # write seqinfo.ini (overwrite to be safe)
    #         max_frame = 0
    #         if hasattr(self, '_last_gt_tracks') and self._last_gt_tracks:
    #             max_frame = max(max_frame, max(self._last_gt_tracks.keys()))
    #         if hasattr(self, '_last_pred_tracks') and self._last_pred_tracks:
    #             max_frame = max(max_frame, max(self._last_pred_tracks.keys()))
    #         with open(gt_seq_dir / 'seqinfo.ini', 'w') as f:
    #             f.write("[Sequence]\n")
    #             f.write(f"name={seq_name}\n")
    #             f.write(f"seqLength={max_frame + 1}\n")
    #             f.write(f"imWidth={self.img_width}\n")
    #             f.write(f"imHeight={self.img_height}\n")
    #             f.write("imExt=.jpg\n")
    #             f.write("imDir=img1\n")
    #             f.write("frameRate=30\n")

    #         # write seqmap in both common places
    #         seqmap_lines = [f"{seq_name}\n"]
    #         with open(Path(abs_gt) / 'seqmap.txt', 'w') as f:
    #             f.writelines(seqmap_lines)
    #         with open(seqmaps_dir / 'seqmap.txt', 'w') as f:
    #             f.writelines(seqmap_lines)

    #         logger.info(f"Created GT and tracker files and seqmap under {temp_dir}")

    #         # Now prepare dataset_config (relative paths, TrackEval expects relative to cwd)
    #         base_dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    #         base_dataset_cfg['GT_FOLDER'] = 'gt'
    #         base_dataset_cfg['TRACKERS_FOLDER'] = 'trackers'
    #         base_dataset_cfg['OUTPUT_FOLDER'] = 'output'
    #         base_dataset_cfg['TRACKERS_TO_EVAL'] = ['tracker']
    #         base_dataset_cfg['CLASSES_TO_EVAL'] = ['pedestrian']
    #         base_dataset_cfg['TRACKER_SUB_FOLDER'] = 'data'
    #         base_dataset_cfg['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/gt.txt'

    #         # Try two modes: first DO_PREPROC=False (skip preproc seqmap checks), then True
    #         for do_preproc in (False, True):
    #             cfg = dict(base_dataset_cfg)
    #             cfg['DO_PREPROC'] = bool(do_preproc)
    #             # force seqmap parameters to standard names
    #             cfg['SEQMAP_FOLDER'] = 'gt'
    #             cfg['SEQMAP_FILE'] = 'seqmap.txt'

    #             # chdir into temp_dir so relative resolution works
    #             original_cwd = os.getcwd()
    #             os.chdir(temp_dir)
    #             try:
    #                 logger.info(f"Attempting TrackEval run with DO_PREPROC={do_preproc}, cwd={os.getcwd()}")
    #                 evaluator = trackeval.Evaluator(eval_config)
    #                 dataset_list = [trackeval.datasets.MotChallenge2DBox(cfg)]
    #                 metrics_list = [trackeval.metrics.HOTA({'THRESHOLD': float(self.iou_threshold)})]
    #                 output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

    #                 # parse result
    #                 if output_res:
    #                     for dataset_name, dataset_res in output_res.items():
    #                         for tracker_name, tracker_res in dataset_res.items():
    #                             for seq_n, seq_res in tracker_res.items():
    #                                 if 'HOTA' in seq_res:
    #                                     hota_data = seq_res['HOTA']
    #                                     hota_value = float(hota_data.get('HOTA', 0.0))
    #                                     deta_value = float(hota_data.get('DetA', 0.0))
    #                                     assa_value = float(hota_data.get('AssA', 0.0))
    #                                     logger.info(f"HOTA computed: {hota_value:.3f} (DO_PREPROC={do_preproc})")
    #                                     return {'hota': hota_value, 'deta': deta_value, 'assa': assa_value}
    #                 logger.warning(f"TrackEval run produced no HOTA with DO_PREPROC={do_preproc}")
    #             except Exception as e:
    #                 logger.warning(f"TrackEval attempt failed with DO_PREPROC={do_preproc}: {e}")
    #                 import traceback
    #                 logger.debug(traceback.format_exc())
    #             finally:
    #                 os.chdir(original_cwd)

    #         logger.error("All TrackEval attempts (DO_PREPROC False/True) failed to produce HOTA.")
    #     except Exception as e:
    #         logger.warning(f"TrackEval outer failure: {e}")
    #         import traceback
    #         logger.debug(traceback.format_exc())

    #     return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}

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
    
    """
    evaluator = TrackingEvaluator(iou_threshold=iou_threshold)
    return evaluator.evaluate(gt_annotation, pred_annotation)
    