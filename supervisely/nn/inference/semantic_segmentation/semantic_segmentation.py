from typing import Any, Dict, List, Optional

import numpy as np

from supervisely.annotation.label import Label
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.inference.inference import Inference
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.sly_logger import logger


class SemanticSegmentation(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "semantic segmentation"
        info["tracking_on_videos_support"] = False
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return Bitmap

    def _create_label(self, dto: PredictionSegmentation, classes_whitelist: Optional[List[str]] = None):
        class_names = self.get_classes()
        if classes_whitelist is not None:
            idx_to_remove = [
                idx
                for idx, _ in enumerate(class_names)
                if class_names[idx] not in classes_whitelist
            ]
            if idx_to_remove:
                logger.debug(
                    f"Classes {idx_to_remove} are not in classes whitelist and will be set to background"
                )
                dto.mask[np.isin(dto.mask, idx_to_remove)] = 0
        image_classes = np.unique(dto.mask)

        labels = []
        for class_idx in image_classes:
            class_mask = dto.mask == class_idx
            class_name = class_names[class_idx]
            obj_class = self.model_meta.get_obj_class(class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {class_name} not found in model classes {class_names}"
                )
            if not class_mask.any():  # skip empty masks
                logger.debug(f"Mask of class {class_name} is empty and will be sklipped")
                return None
            geometry = Bitmap(class_mask, extra_validation=False)
            label = Label(geometry, obj_class)
            labels.append(label)
        return labels

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionSegmentation]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[PredictionSegmentation]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )
