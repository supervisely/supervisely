import numpy as np


def calculate_mean_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> float:
    """
    Calculate mean IoU per class for semantic segmentation.
    
    Only classes present in ground truth are considered for mean calculation.
    Background (class 0) and ignore_index pixels are excluded.
    
    Args:
        pred_mask: Predicted segmentation mask (H, W) with class indices
        gt_mask: Ground truth segmentation mask (H, W) with class indices
        num_classes: Total number of classes including background
        ignore_index: Index to ignore in evaluation (default 255)
        
    Returns:
        Mean IoU across classes present in GT (excluding background)
        Returns 0.0 if no valid classes found
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}"
        )
    
    # Create mask for valid pixels (exclude ignore_index)
    valid_mask = (gt_mask != ignore_index)
    
    # Apply valid mask
    pred_valid = pred_mask[valid_mask]
    gt_valid = gt_mask[valid_mask]
    
    # Calculate per-class IoU
    ious = []
    
    for class_idx in range(1, num_classes):  # Skip background (0)
        # Check if class exists in GT
        gt_class_mask = (gt_valid == class_idx)
        
        if not gt_class_mask.any():
            continue  # Skip classes not present in GT
        
        pred_class_mask = (pred_valid == class_idx)
        
        # Calculate intersection and union
        intersection = np.logical_and(gt_class_mask, pred_class_mask).sum()
        union = np.logical_or(gt_class_mask, pred_class_mask).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        ious.append(iou)
    
    # Return mean IoU
    if len(ious) == 0:
        return 0.0
    
    return float(np.mean(ious))