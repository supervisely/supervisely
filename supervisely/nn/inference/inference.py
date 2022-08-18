import os
from re import L
from typing import List
from supervisely._utils import is_production, is_development
from supervisely.app.fastapi.subapp import get_name_from_env
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.annotation.tag import Tag
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.geometry.bitmap import Bitmap
from supervisely.project.project_meta import ProjectMeta
from supervisely.nn.prediction_dto import PredictionMask
from supervisely.sly_logger import logger
import supervisely.imaging.image as sly_image


class Inference:
    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir
        self._model_meta = None
        self._confidence = "confidence"
        if is_production():
            raise NotImplementedError("TBD - download directory")
        elif is_development():
            pass

    def get_classes(self) -> List[str]:
        raise NotImplementedError(
            "Have to be implemented in child class after inheritance"
        )

    def get_info(self) -> dict:
        return {
            "model name": get_name_from_env(default="Neural Network Serving"),
            "model dir": self.model_dir,
            "number of classes": len(self.get_classes()),
        }

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            classes = []
            for name in self.get_classes():
                classes.append(ObjClass(name, Bitmap))
            self._model_meta = ProjectMeta(classes)
            self.get_confidence_tag_meta()  # @TODO: optimize, create if needed
        return self._model_meta

    def get_confidence_tag_meta(self):
        tag_meta = self.model_meta.get_tag_meta(self._confidence)
        if tag_meta is None:
            tag_meta = TagMeta(self._confidence, TagValueType.ANY_NUMBER)
            self._model_meta = self._model_meta.add_tag_meta(tag_meta)
        return tag_meta

    def get_inference_settings(self):
        return {}

    def _create_label(self, dto) -> Label:
        raise NotImplementedError("Have to be implemented in child class")

    def predict(self, image_path: str):
        raise NotImplementedError("Have to be implemented in child class")

    def predict_annotation(self, image_path: str) -> Annotation:
        predictions = self.predict(image_path)
        return self._predictions_to_annotation(image_path, predictions)

    def _predictions_to_annotation(
        self, image_path: str, predictions: List
    ) -> Annotation:
        labels = []
        for prediction in predictions:
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            labels.append(label)

        # create annotation with correct image resolution
        ann = Annotation.from_img_path(image_path)
        ann = ann.add_labels(labels)
        return ann


class InstanceSegmentation(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "instance segmentation"
        return info

    def _create_label(self, dto: PredictionMask):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        if not dto.mask.any():  # skip empty masks
            logger.debug(
                f"Mask of class {dto.class_name} is empty and will be sklipped"
            )
            return None
        geometry = Bitmap(dto.mask)
        tags = []
        if dto.score is not None:
            tags.append(Tag(self.get_confidence_tag_meta(), dto.score))
        label = Label(geometry, obj_class, tags)
        return label

    def predict(self, image_path: str) -> list[PredictionMask]:
        raise NotImplementedError("Have to be implemented in child class")

    def visualize(
        self, predictions: list[PredictionMask], image_path: str, vis_path: str
    ):
        image = sly_image.read(image_path)
        ann = self._predictions_to_annotation(image_path, predictions)
        ann.draw_pretty(bitmap=image, thickness=2, output_path=vis_path)
