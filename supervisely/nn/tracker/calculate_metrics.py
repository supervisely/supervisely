import numpy as np
from collections import defaultdict
from typing import Dict, List, Union

from scipy.optimize import linear_sum_assignment  # pylint: disable=import-error

import supervisely as sly
from supervisely.video_annotation.video_annotation import VideoAnnotation

import motmetrics as mm  # pylint: disable=import-error

class TrackingEvaluator:
    """
    Evaluator for video tracking metrics including MOTA, MOTP, IDF1.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """Initialize evaluator with IoU threshold for matching."""
        from supervisely.nn.tracker import TRACKING_LIBS_INSTALLED
        if not TRACKING_LIBS_INSTALLED:
            raise ImportError(
                "Tracking dependencies are not installed. "
                "Please install supervisely with `pip install supervisely[tracking]`."
            )
            
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_annotation: VideoAnnotation,
        pred_annotation: VideoAnnotation,
    ) -> Dict[str, Union[float, int]]:
        """Main entry: extract tracks from annotations, compute basic and MOT metrics, return results."""
        self._validate_annotations(gt_annotation, pred_annotation)
        self.img_height, self.img_width = gt_annotation.img_size

        gt_tracks = self._extract_tracks(gt_annotation)
        pred_tracks = self._extract_tracks(pred_annotation)

        basic = self._compute_basic_metrics(gt_tracks, pred_tracks)
        mot = self._compute_mot_metrics(gt_tracks, pred_tracks)

        results = {
            # basic detection
            "precision": basic["precision"],
            "recall": basic["recall"],
            "f1": basic["f1"],
            "avg_iou": basic["avg_iou"],
            "true_positives": basic["tp"],
            "false_positives": basic["fp"],
            "false_negatives": basic["fn"],
            "total_gt_objects": basic["total_gt"],
            "total_pred_objects": basic["total_pred"],

            # motmetrics
            "mota": mot["mota"],
            "motp": mot["motp"],
            "idf1": mot["idf1"],
            "id_switches": mot["id_switches"],
            "fragmentations": mot["fragmentations"],
            "num_misses": mot["num_misses"],
            "num_false_positives": mot["num_false_positives"],

            # config
            "iou_threshold": self.iou_threshold,
        }
        return results

    def _validate_annotations(self, gt: VideoAnnotation, pred: VideoAnnotation):
        """Minimal type validation for annotations."""
        if not isinstance(gt, VideoAnnotation) or not isinstance(pred, VideoAnnotation):
            raise TypeError("gt_annotation and pred_annotation must be VideoAnnotation instances")

    def _extract_tracks(self, annotation: VideoAnnotation) -> Dict[int, List[Dict]]:
        """
        Extract tracks from a VideoAnnotation into a dict keyed by frame index.
        Each element is a dict: {'track_id': int, 'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_name': str}
        """
        frames_to_tracks = defaultdict(list)

        for frame in annotation.frames:
            frame_idx = frame.index
            for figure in frame.figures:
                # use track_id if present, otherwise fallback to object's key int
                track_id = int(figure.track_id) if figure.track_id is not None else figure.video_object.key().int

                bbox = figure.geometry
                if not isinstance(bbox, sly.Rectangle):
                    bbox = bbox.to_bbox()

                x1 = float(bbox.left)
                y1 = float(bbox.top)
                x2 = float(bbox.right)
                y2 = float(bbox.bottom)

                frames_to_tracks[frame_idx].append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(getattr(figure, "confidence", 1.0)),
                    "class_name": figure.video_object.obj_class.name
                })

        return dict(frames_to_tracks)

    def _compute_basic_metrics(self, gt_tracks: Dict[int, List[Dict]], pred_tracks: Dict[int, List[Dict]]):
        """
        Compute per-frame true positives / false positives / false negatives and average IoU.
        Matching is performed with Hungarian algorithm (scipy). Matches with IoU < threshold are discarded.
        """
        tp = fp = fn = 0
        total_iou = 0.0
        iou_count = 0

        frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        for f in frames:
            gts = gt_tracks.get(f, [])
            preds = pred_tracks.get(f, [])

            if not gts and not preds:
                continue
            if not gts:
                fp += len(preds)
                continue
            if not preds:
                fn += len(gts)
                continue

            gt_boxes = np.array([g["bbox"] for g in gts])
            pred_boxes = np.array([p["bbox"] for p in preds])

            # get cost matrix from motmetrics (cost = 1 - IoU)
            cost_mat = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
            # replace NaNs (if any) with a large cost so Hungarian will avoid them
            cost_for_assignment = np.where(np.isnan(cost_mat), 1e6, cost_mat)

            # Hungarian assignment (minimize cost -> maximize IoU)
            row_idx, col_idx = linear_sum_assignment(cost_for_assignment)

            matched_gt = set()
            matched_pred = set()
            for r, c in zip(row_idx, col_idx):
                if r < cost_mat.shape[0] and c < cost_mat.shape[1]:
                    # IoU = 1 - cost
                    cost_val = cost_mat[r, c]
                    if np.isnan(cost_val):
                        continue
                    iou_val = 1.0 - float(cost_val)
                    if iou_val >= self.iou_threshold:
                        matched_gt.add(r)
                        matched_pred.add(c)
                        total_iou += iou_val
                        iou_count += 1

            frame_tp = len(matched_gt)
            frame_fp = len(preds) - len(matched_pred)
            frame_fn = len(gts) - len(matched_gt)

            tp += frame_tp
            fp += frame_fp
            fn += frame_fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = total_iou / iou_count if iou_count > 0 else 0.0

        total_gt = sum(len(v) for v in gt_tracks.values())
        total_pred = sum(len(v) for v in pred_tracks.values())

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_iou": avg_iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_gt": total_gt,
            "total_pred": total_pred,
        }

    def _compute_mot_metrics(self, gt_tracks: Dict[int, List[Dict]], pred_tracks: Dict[int, List[Dict]]):
        """
        Use motmetrics.MOTAccumulator to collect associations per frame and compute common MOT metrics.
        Distance matrix is taken directly from motmetrics.distances.iou_matrix (which returns 1 - IoU).
        Pairs with distance > (1 - iou_threshold) are set to infinity to exclude them from matching.
        """
        acc = mm.MOTAccumulator(auto_id=True)

        frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        for f in frames:
            gts = gt_tracks.get(f, [])
            preds = pred_tracks.get(f, [])

            gt_ids = [g["track_id"] for g in gts]
            pred_ids = [p["track_id"] for p in preds]

            if gts and preds:
                gt_boxes = np.array([g["bbox"] for g in gts])
                pred_boxes = np.array([p["bbox"] for p in preds])

                # motmetrics provides a distance matrix (1 - IoU)
                dist_mat = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
                # exclude pairs with IoU < threshold => distance > 1 - threshold
                dist_mat = np.array(dist_mat, dtype=float)
                dist_mat[np.isnan(dist_mat)] = np.inf
                dist_mat[dist_mat > (1.0 - self.iou_threshold)] = np.inf
            else:
                dist_mat = np.full((len(gts), len(preds)), np.inf)

            acc.update(gt_ids, pred_ids, dist_mat)

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                "mota",
                "motp",
                "idf1",
                "num_switches",
                "num_fragmentations",
                "num_misses",
                "num_false_positives",
            ],
            name="eval",
        )

        def get_val(col: str, default=0.0):
            if summary.empty or col not in summary.columns:
                return float(default)
            v = summary.iloc[0][col]
            return float(v) if not np.isnan(v) else float(default)

        return {
            "mota": get_val("mota", 0.0),
            "motp": get_val("motp", 0.0),
            "idf1": get_val("idf1", 0.0),
            "id_switches": int(get_val("num_switches", 0.0)),
            "fragmentations": int(get_val("num_fragmentations", 0.0)),
            "num_misses": int(get_val("num_misses", 0.0)),
            "num_false_positives": int(get_val("num_false_positives", 0.0)),
        }


def evaluate(
    gt_annotation: VideoAnnotation,
    pred_annotation: VideoAnnotation,
    iou_threshold: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Evaluate tracking predictions against ground truth.

    Args:
        gt_annotation: Ground-truth annotation, an object of class supervisely VideoAnnotation containing reference object tracks.
        pred_annotation: Predicted annotation, an object of class supervisely VideoAnnotation to be compared against the ground truth.
        iou_threshold: Minimum Intersection-over-Union required for a detection to be considered a valid match.

    Returns:
        dict: json with evaluation metrics.
    """
    evaluator = TrackingEvaluator(iou_threshold=iou_threshold)
    return evaluator.evaluate(gt_annotation, pred_annotation)
