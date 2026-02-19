import numpy as np
import cv2

class SegmentationMetrics:
    """Computes segmentation metrics: mIoU, boundary IoU, pixel accuracy with optional ignore index."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        boundary_dilation_ratio: float = 0.02,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_dilation_ratio = boundary_dilation_ratio

    def _boundary_iou(self, gt: np.ndarray, pred: np.ndarray) -> float:
        gt_b = self._mask_to_boundary(gt)
        pred_b = self._mask_to_boundary(pred)

        inter = ((gt_b & pred_b) > 0).sum()
        union = ((gt_b | pred_b) > 0).sum()

        return inter / union if union > 0 else 0.0

    def _mask_to_boundary(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        diag = np.sqrt(h * h + w * w)
        dilation = max(1, int(round(self.boundary_dilation_ratio * diag)))

        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(padded, kernel, iterations=dilation)
        eroded = eroded[1 : h + 1, 1 : w + 1]

        return mask - eroded

    def all_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        exclude_background: bool = True,
    ) -> dict:
        if pred.shape != gt.shape:
            raise ValueError("Shape mismatch")

        valid = gt != self.ignore_index
        start = 1 if exclude_background else 0

        per_class_iou = {}
        per_class_boundary_iou = {}

        for cls in range(start, self.num_classes):
            gt_cls = (gt == cls) & valid
            if not gt_cls.any():
                continue

            pred_cls = (pred == cls) & valid

            gt_bin = gt_cls.astype(np.uint8)
            pred_bin = pred_cls.astype(np.uint8)

            inter = np.logical_and(gt_cls, pred_cls).sum()
            union = np.logical_or(gt_cls, pred_cls).sum()
            per_class_iou[cls] = inter / union if union > 0 else 0.0

            per_class_boundary_iou[cls] = self._boundary_iou(gt_bin, pred_bin)

        mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0
        mean_boundary_iou = (
            float(np.mean(list(per_class_boundary_iou.values())))
            if per_class_boundary_iou else 0.0
        )

        return {
            "mean_iou": mean_iou,
            "mean_boundary_iou": mean_boundary_iou,
            "per_class_iou": per_class_iou,
            "per_class_boundary_iou": per_class_boundary_iou,
        }