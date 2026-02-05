import numpy as np
import cv2

class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        boundary_dilation_ratio: float = 0.02,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_dilation_ratio = boundary_dilation_ratio

    def calculate_mean_iou(
            self, 
            pred_mask: np.ndarray, 
            gt_mask: np.ndarray, 
            exclude_background: 
            bool = True
            ) -> float:
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
   
    def calculate_mean_boundary_iou(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        exclude_background: bool = True,
        ) -> float:
        if pred.shape != gt.shape:
            raise ValueError("Shape mismatch")

        valid = gt != self.ignore_index

        bious = []
        start = 1 if exclude_background else 0

        for cls in range(start, self.num_classes):
            gt_cls = (gt == cls) & valid
            if not gt_cls.any():
                continue

            pred_cls = (pred == cls) & valid

            gt_bin = gt_cls.astype(np.uint8)
            pred_bin = pred_cls.astype(np.uint8)

            bious.append(self._boundary_iou(gt_bin, pred_bin))

        return float(np.mean(bious)) if bious else 0.0

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