from typing import Any, Dict, List

import numpy as np

from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.inference.inference import Inference
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.project.project_meta import ProjectMeta
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

    def _find_bg_class_index(self, class_names: List[str]):
        possible_bg_names = ["background", "bg", "unlabeled", "neutral", "__bg__"]
        bg_class_index = None
        for i, name in enumerate(class_names):
            if name in possible_bg_names:
                bg_class_index = i
                break
        return bg_class_index

    def _add_default_bg_class(self, meta: ProjectMeta):
        default_bg_class_name = "__bg__"
        obj_class = meta.get_obj_class(default_bg_class_name)
        if obj_class is None:
            obj_class = ObjClass(default_bg_class_name, self._get_obj_class_shape())
            meta = meta.add_obj_class(obj_class)
        return meta, obj_class

    def _get_or_create_bg_obj_class(self, classes):
        bg_class_index = self._find_bg_class_index(classes)
        if bg_class_index is None:
            self._model_meta, bg_obj_class = self._add_default_bg_class(self.model_meta)
        else:
            bg_class_name = classes[bg_class_index]
            bg_obj_class = self.model_meta.get_obj_class(bg_class_name)
        return bg_obj_class

    def _create_label(self, dto: PredictionSegmentation, classes_whitelist: List[str] = None):
        classes = self.get_classes()

        image_classes_indexes = np.unique(dto.mask)
        labels = []
        for class_idx in image_classes_indexes:
            class_mask = dto.mask == class_idx
            class_name = classes[class_idx]
            if classes_whitelist not in (None, "all") and class_name not in classes_whitelist:
                obj_class = self._get_or_create_bg_obj_class(classes)
            else:
                obj_class = self.model_meta.get_obj_class(class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {class_name} not found in model classes {self.get_classes()}"
                )
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
