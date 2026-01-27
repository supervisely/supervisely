import cv2
import numpy as np
from typing import Dict, Optional, Any, Tuple

import supervisely as sly
from supervisely import logger
from supervisely.nn import TaskType

from .metrics import calculate_mean_iou
# from .metrics import calculate_detection_metrics

class Evaluator:
    """
    App-agnostic evaluator for Live Training.
    
    Stores predictions (sly objects) by image_id, converts them to task-specific format,
    computes metrics when ground truth arrives, and tracks values using EMA.
    """
    
    def __init__(
        self,
        task_type: str,
        class2idx: dict,
        ema_alpha: float = 0.1,
        ignore_index: int = 255
    ):
        """
        Initialize evaluator.
        
        Args:
            task_type: Type of task (TaskType.SEMANTIC_SEGMENTATION, etc.)
            class2idx: Mapping from class name to class index
            ema_alpha: EMA smoothing factor (0-1), lower = more stable
            ignore_index: Index to ignore in evaluation (for segmentation)
        """
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        
        self.task_type = task_type
        self.class2idx = class2idx
        self.ema_alpha = ema_alpha
        self.ignore_index = ignore_index
        
        # Storage for predictions: {image_id: (objects, image_shape)}
        self._predictions: Dict[Any, Tuple[list, tuple]] = {}
        
        # EMA tracking
        self.ema_value: Optional[float] = None
        self.sample_count: int = 0
    
    def store_prediction(self, image_id: Any, objects: list, image_shape: tuple):
        """
        Store prediction (sly objects) for later evaluation.
        
        Args:
            image_id: Unique identifier for the image
            objects: List of Supervisely label objects (json format)
            image_shape: (height, width) of the image
        """
        if image_id in self._predictions:
            logger.warning(
                f"Prediction for image_id={image_id} already exists, overwriting"
            )
        
        self._predictions[image_id] = (objects, image_shape)
    
    def evaluate(
        self,
        image_id: Any,
        ground_truth_annotation: sly.Annotation
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate stored prediction against ground truth annotation.
        
        Args:
            image_id: Unique identifier for the image
            ground_truth_annotation: Supervisely Annotation object (GT)
            
        Returns:
            Dictionary with 'metric_value' and 'ema_value', or None if no prediction stored
        """
        if image_id not in self._predictions:
            logger.warning(
                f"No prediction stored for image_id={image_id}. "
                f"This can happen if predict was not called before add_sample. "
                f"Skipping evaluation for this image."
            )
            return None
        
        objects, image_shape = self._predictions[image_id]
        
        # Convert to task-specific format and calculate metric
        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            pred_mask = self._objects_to_mask(objects, image_shape)
            gt_mask = self._annotation_to_mask(ground_truth_annotation, image_shape)
            metric_value = calculate_mean_iou(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                num_classes=len(self.class2idx),
                ignore_index=self.ignore_index
            )
        
        elif self.task_type == TaskType.OBJECT_DETECTION:
            # TODO: implement when detection is needed
            pred_bboxes = self._objects_to_bboxes(objects)
            gt_bboxes = self._annotation_to_bboxes(ground_truth_annotation)
            # metric_value = calculate_detection_metrics(
            #     pred_bboxes=pred_bboxes,
            #     gt_bboxes=gt_bboxes
            # )
        
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        
        # Update EMA
        self._update_ema(metric_value)
        
        # Clean up prediction to save memory
        del self._predictions[image_id]
        
        return {
            'metric_value': metric_value,
            'ema_value': self.ema_value
        }
    
    def _objects_to_mask(self, objects: list, image_shape: tuple) -> np.ndarray:
        """
        Convert Supervisely objects (json format) to segmentation mask.
        
        Args:
            objects: List of label objects in json format
            image_shape: (height, width)
            
        Returns:
            Numpy array (H, W) with class indices
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create minimal ProjectMeta for parsing
        project_meta = self._get_project_meta_stub()
        
        # Parse each object and draw on mask
        for obj in objects:
            # Get class name
            obj_class_title = obj.get('classTitle')
            if obj_class_title is None:
                continue
            
            # Map to index
            class_idx = self.class2idx.get(obj_class_title)
            if class_idx is None:
                continue
            
            # Parse Label from json
            try:
                label = sly.Label.from_json(obj, project_meta)
                # Draw geometry on mask
                label.geometry.draw(mask, color=class_idx)
            except Exception as e:
                logger.warning(f"Failed to parse object geometry: {e}")
                continue
        
        return mask
    
    def _annotation_to_mask(
        self,
        annotation: sly.Annotation,
        image_shape: tuple
    ) -> np.ndarray:
        """
        Convert Supervisely Annotation to segmentation mask.
        
        Args:
            annotation: Supervisely Annotation object
            image_shape: (height, width)
            
        Returns:
            Numpy array (H, W) with class indices
        """
        height, width = image_shape
        
        # Convert to non-overlapping masks
        mapping = {label.obj_class: label.obj_class for label in annotation.labels}
        ann_nonoverlap = annotation.to_nonoverlapping_masks(mapping)
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw each label
        for label in ann_nonoverlap.labels:
            class_name = label.obj_class.name
            class_idx = self.class2idx.get(class_name)
            
            if class_idx is not None:
                label.geometry.draw(mask, color=class_idx)
        
        return mask
    
    def _get_project_meta_stub(self) -> sly.ProjectMeta:
        """Create minimal ProjectMeta for parsing geometries."""
        obj_classes = [
            sly.ObjClass(class_name, sly.AnyGeometry)
            for class_name in self.class2idx.keys()
        ]
        return sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
    
    def _objects_to_bboxes(self, objects: list) -> dict:
        """Convert Supervisely objects to detection format. TODO: Implement."""
        raise NotImplementedError("Detection support not yet implemented")
    
    def _annotation_to_bboxes(self, annotation: sly.Annotation) -> dict:
        """Convert Supervisely Annotation to detection format. TODO: Implement."""
        raise NotImplementedError("Detection support not yet implemented")
    
    def _update_ema(self, new_value: float):
        """Update EMA with new metric value."""
        self.sample_count += 1
        
        if self.ema_value is None:
            self.ema_value = new_value
        else:
            self.ema_value = (
                self.ema_alpha * new_value + 
                (1 - self.ema_alpha) * self.ema_value
            )
    
    def has_pending_prediction(self, image_id: Any) -> bool:
        """Check if prediction is stored for given image_id."""
        return image_id in self._predictions
    
    def clear_prediction(self, image_id: Any):
        """Clear stored prediction without evaluation."""
        if image_id in self._predictions:
            del self._predictions[image_id]
    
    def reset(self):
        """Reset evaluator state (EMA, counters, stored predictions)."""
        self._predictions.clear()
        self.ema_value = None
        self.sample_count = 0
    
    def state_dict(self) -> Dict:
        """Get evaluator state for checkpointing."""
        return {
            'task_type': self.task_type,
            'ema_alpha': self.ema_alpha,
            'ema_value': self.ema_value,
            'sample_count': self.sample_count,
            'ignore_index': self.ignore_index
        }
    
    def load_state_dict(self, state: Dict):
        """Load evaluator state from checkpoint."""
        self.ema_value = state.get('ema_value')
        self.sample_count = state.get('sample_count', 0)