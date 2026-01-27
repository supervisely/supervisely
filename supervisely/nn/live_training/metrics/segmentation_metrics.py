import numpy as np
from typing import Optional

class SegmentationMetrics:
    """
    Class for calculating segmentation metrics.

    Currently implements mean IoU. Can be extended with other metrics or
    accumulation over batches.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def calculate_mean_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray, exclude_background: bool = True) -> float:
        """
        Calculate mean IoU for a single pair of masks.
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}")

        valid_mask = gt_mask != self.ignore_index
        pred_valid = pred_mask[valid_mask]
        gt_valid = gt_mask[valid_mask]

        ious = []
        start_idx = 1 if exclude_background else 0
        for class_idx in range(start_idx, self.num_classes):
            gt_class = gt_valid == class_idx
            if not gt_class.any():
                continue
            pred_class = pred_valid == class_idx
            intersection = np.logical_and(gt_class, pred_class).sum()
            union = np.logical_or(gt_class, pred_class).sum()
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

        return float(np.mean(ious)) if ious else 0.0
