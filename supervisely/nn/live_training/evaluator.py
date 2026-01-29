import numpy as np
from typing import Dict, Optional, Any, Tuple, Mapping

import supervisely as sly
from supervisely import logger
from supervisely.nn import TaskType

from .metrics import SegmentationMetrics, DetectionMetrics

class Evaluator:
    """
    App-agnostic evaluator for Live Training.

    Stores predictions (sly objects) by image_id, converts them to task-specific format,
    computes metrics when ground truth arrives, and tracks values using EMA.
    """

    def __init__(
        self,
        task_type: TaskType,
        class2idx: Mapping[str, int],
        ema_alpha: float = 0.1,
        ignore_index: int = 255,
        score_thr: float | None = None,
    ):
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self.task_type: TaskType = task_type
        self.class2idx: Mapping[str, int] = class2idx
        self.ema_alpha: float = ema_alpha
        self.ignore_index: int = ignore_index
        self.score_thr: float = score_thr

        self._predictions: Dict[Any, Tuple[list, Tuple[int, int]]] = {}

        self.ema_value: Optional[float] = None

    def store_prediction(self, image_id: Any, objects: list, image_shape: Tuple[int, int]):
        if image_id in self._predictions:
            logger.warning(f"Prediction for image_id={image_id} already exists, overwriting")
        self._predictions[image_id] = (objects, image_shape)

    def evaluate(
        self, image_id: Any, ground_truth_annotation: sly.Annotation
    ) -> Optional[Dict[str, float]]:
        if image_id not in self._predictions:
            logger.warning(f"No prediction stored for image_id={image_id}. Skipping evaluation for this image")
            return None

        objects, image_shape = self._predictions.pop(image_id)  

        metric_value: Optional[float] = None
        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            metric_value = self._evaluate_segmentation(objects, ground_truth_annotation, image_shape)
        elif self.task_type == TaskType.OBJECT_DETECTION:
            metric_value = self._evaluate_detection(objects, ground_truth_annotation, image_shape)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        if metric_value is not None:
            self._update_ema(metric_value)

        return {'metric_value': metric_value, 'ema_value': self.ema_value}

    def _evaluate_segmentation(self, objects: list, gt_annotation: sly.Annotation, image_shape: Tuple[int,int]) -> float:
        pred_mask = self._pred_objects_to_mask(objects, image_shape)
        gt_mask = self._gt_annotation_to_mask(gt_annotation, image_shape)
        metrics = SegmentationMetrics(num_classes=len(self.class2idx), ignore_index=self.ignore_index)
        return metrics.calculate_mean_iou(pred_mask, gt_mask)

    def _evaluate_detection(self, objects: list, gt_annotation: sly.Annotation, image_shape: Tuple[int,int]) -> float:
        if self.score_thr is not None:
            objects = [obj for obj in objects if obj.get('meta', {}).get('confidence') >= self.score_thr]
        pred_boxes, pred_labels = self._pred_objects_to_bboxes(objects)
        gt_boxes, gt_labels = self._gt_annotation_to_bboxes(gt_annotation)
        metrics_calc = DetectionMetrics()
        metrics = metrics_calc.all_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels)
        return metrics['assistance_score']
    
    def _pred_objects_to_mask(self, objects: list, image_shape: Tuple[int,int]) -> np.ndarray:
        """Convert predicted objects to segmentation mask."""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.int32)

        project_meta = self._get_project_meta_stub()
        for obj in objects:
            obj_class_title = obj['classTitle']
            if obj_class_title is None:
                continue

            class_idx = self.class2idx.get(obj_class_title)
            if class_idx is None:
                continue

            try:
                label = sly.Label.from_json(obj, project_meta)
                label.geometry.draw(mask, color=class_idx)
            except Exception as e:
                logger.warning(f"Failed to parse object geometry: {e}")
        return mask

    def _gt_annotation_to_mask(self, annotation: sly.Annotation, image_shape: Tuple[int,int]) -> np.ndarray:
        """Convert ground truth annotation to segmentation mask."""
        height, width = image_shape
        mapping = {label.obj_class: label.obj_class for label in annotation.labels}
        ann_nonoverlap = annotation.to_nonoverlapping_masks(mapping)

        mask = np.zeros((height, width), dtype=np.int32)
        for label in ann_nonoverlap.labels:
            class_name = label.obj_class.name
            class_idx = self.class2idx.get(class_name)
            if class_idx is not None:
                try:
                    label.geometry.draw(mask, color=class_idx)
                except Exception as e:
                    logger.warning(f"Failed to draw label geometry: {e}")
        return mask

    def _pred_objects_to_bboxes(self, objects: list):
        bboxes = []
        labels = []

        project_meta = self._get_project_meta_stub()

        for obj in objects:
            class_title = obj.get('classTitle')
            if class_title is None:
                continue

            class_idx = self.class2idx.get(class_title)
            if class_idx is None:
                continue

            try:
                label = sly.Label.from_json(obj, project_meta)
                geom = label.geometry
                if not isinstance(geom, sly.Rectangle):
                    continue  

                bbox = [geom.left, geom.top, geom.right, geom.bottom]
                bboxes.append(bbox)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Failed to parse prediction bbox: {e}")

        return (np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)) 


    def _gt_annotation_to_bboxes(self, annotation: sly.Annotation):
        bboxes = []
        labels = []

        for label in annotation.labels:
            class_idx = self.class2idx.get(label.obj_class.name)
            if class_idx is None:
                continue
            geom = label.geometry
            if not isinstance(geom, sly.Rectangle):
                continue

            bboxes.append([geom.left, geom.top, geom.right, geom.bottom])
            labels.append(class_idx)

        return (np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64))

    
    def _get_project_meta_stub(self) -> sly.ProjectMeta:
        obj_classes = [sly.ObjClass(class_name, sly.AnyGeometry) for class_name in self.class2idx.keys()]
        return sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))

    def _update_ema(self, new_value: float):
        if self.ema_value is None:
            self.ema_value = new_value
        else:
            self.ema_value = self.ema_alpha * new_value + (1 - self.ema_alpha) * self.ema_value

    def reset(self):
        self._predictions.clear()
        self.ema_value = None

    def state_dict(self) -> Dict:
        return {
            'task_type': self.task_type,
            'ema_alpha': self.ema_alpha,
            'ema_value': self.ema_value,
            'ignore_index': self.ignore_index
        }

    def load_state_dict(self, state: Dict):
        self.ema_value = state.get('ema_value')