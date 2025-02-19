from typing import Any, Dict, List

from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.nn.inference.inference import Inference
from supervisely.nn.prediction_dto import PredictionAlphaMask
from supervisely.sly_logger import logger


class InstanceProbabilitySegmentation(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "instance segmentation"
        return info

    def _get_obj_class_shape(self):
        return AlphaMask

    def _create_label(self, dto: PredictionAlphaMask):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(f"Class {dto.class_name} not found in model classes {self.classes}")

        if not dto.mask.any():  # skip empty masks
            logger.debug(f"Mask of class {dto.class_name} is empty and will be skipped")
            return None

        geometry = AlphaMask(dto.mask, extra_validation=False)
        tags = None # if dto.score is None else [Tag(self._get_confidence_tag_meta(), dto.score)]
        return Label(geometry, obj_class, tags)

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionAlphaMask]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionAlphaMask]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )
