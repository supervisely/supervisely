import numpy as np
from typing import Dict, Tuple

class DetectionMetrics:
    """Detection metrics for Evaluator (IoU, precision, recall, F1, assistance score)."""

    def __init__(self, iou_min: float = 0.7, iou_perfect: float = 0.9):
        self.iou_min = iou_min
        self.iou_perfect = iou_perfect
        self.score_weights = {
            'tp': 1.0,
            'fs': 0.5,
            'fc': 0.5,
            'fp': -0.5,
            'fn': 0.0
        }

    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = np.array(boxes)
        xywh = np.zeros_like(boxes, dtype=np.float32)
        xywh[:, 0] = boxes[:, 0]
        xywh[:, 1] = boxes[:, 1]
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return xywh

    def iou_matrix(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between predicted and GT boxes."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return np.zeros((len(pred_boxes), len(gt_boxes)))
        
        pred_xywh = self._xyxy_to_xywh(pred_boxes)
        gt_xywh = self._xyxy_to_xywh(gt_boxes)
        
        import pycocotools.mask as mask_utils 
        return mask_utils.iou(pred_xywh, gt_xywh, [0]*len(gt_xywh))

    def match_boxes(self, pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold: float):
        """Match predictions to GT boxes using Hungarian algorithm."""
        from scipy.optimize import linear_sum_assignment

        pred_matches = {}
        gt_matches = {}
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return pred_matches, gt_matches
        
        iou_mat = self.iou_matrix(pred_boxes, gt_boxes)
        valid = iou_mat >= iou_threshold
        cost_mat = -iou_mat.copy()
        cost_mat[~valid] = 0
        
        pred_idx, gt_idx = linear_sum_assignment(cost_mat)
        for p, g in zip(pred_idx, gt_idx):
            iou = iou_mat[p, g]
            if iou >= iou_threshold:
                class_match = pred_labels[p] == gt_labels[g]
                pred_matches[p] = (g, iou, class_match)
                gt_matches[g] = (p, iou, class_match)
        
        return pred_matches, gt_matches

    def assistance_score(self, pred_boxes, pred_labels, gt_boxes, gt_labels) -> Dict:
        """Compute annotation assistance score and components."""
        num_gt = len(gt_boxes)
        if num_gt == 0:
            return {'assistance_score': 0.0, 'tp':0, 'fs':0, 'fc':0, 'fp':0, 'fn':0, 'num_gt':0}

        pred_matches, gt_matches = self.match_boxes(pred_boxes, pred_labels, gt_boxes, gt_labels, self.iou_min)
        
        tp = fs = fc = fp = 0
        for i in range(len(pred_boxes)):
            if i in pred_matches:
                g_idx, iou, class_match = pred_matches[i]
                if iou >= self.iou_perfect and class_match: tp += 1
                elif self.iou_min <= iou < self.iou_perfect and class_match: fs += 1
                elif iou >= self.iou_perfect and not class_match: fc += 1
                else: fp += 1
            else:
                fp += 1
        fn = num_gt - len(gt_matches)
        score = (tp*self.score_weights['tp'] + fs*self.score_weights['fs'] +
                 fc*self.score_weights['fc'] + fp*self.score_weights['fp'] +
                 fn*self.score_weights['fn']) / num_gt

        return {'assistance_score': score, 'tp': tp, 'fs': fs, 'fc': fc, 'fp': fp, 'fn': fn, 'num_gt': num_gt}

    def detection_metrics(self, pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold: float) -> Dict:
        """Compute precision, recall, F1 at given IoU threshold."""
        pred_matches, gt_matches = self.match_boxes(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)
        tp = sum(1 for _, (_, _, c) in pred_matches.items() if c)
        precision = tp / len(pred_boxes) if len(pred_boxes)>0 else 0.0
        recall = tp / len(gt_boxes) if len(gt_boxes)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1}

    def all_metrics(self, pred_boxes, pred_labels, gt_boxes, gt_labels) -> Dict:
        """Compute all metrics together (assistance + detection)."""
        metrics = self.assistance_score(pred_boxes, pred_labels, gt_boxes, gt_labels)
        metrics_min = self.detection_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, self.iou_min)
        metrics_perf = self.detection_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, self.iou_perfect)

        metrics.update({
            f'precision_iou_{self.iou_min}': metrics_min['precision'],
            f'recall_iou_{self.iou_min}': metrics_min['recall'],
            f'f1_score_iou_{self.iou_min}': metrics_min['f1_score'],
            f'precision_iou_{self.iou_perfect}': metrics_perf['precision'],
            f'recall_iou_{self.iou_perfect}': metrics_perf['recall'],
            f'f1_score_iou_{self.iou_perfect}': metrics_perf['f1_score']
        })
        return metrics
