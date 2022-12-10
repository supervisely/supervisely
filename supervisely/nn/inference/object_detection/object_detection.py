from typing import Dict, List, Any
from pathlib import Path
import os
from supervisely.app.widgets.widget import Widget
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.prediction_dto import PredictionBBox
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
import supervisely.imaging.image as sly_image
from supervisely.sly_logger import logger
from supervisely.nn.inference.inference import Inference
from supervisely.annotation.annotation import Annotation
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import Progress


class ObjectDetection(Inference):
    def _get_templates_dir(self):
        # template_dir = os.path.join(
        #     Path(__file__).parent.absolute(), "dashboard/templates"
        # )
        # return template_dir
        return None

    def _get_layout(self) -> Widget:
        return None
        # import supervisely.nn.inference.instance_segmentation.dashboard.main_ui as main_ui
        # return main_ui.menu

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "object detection"
        info["sliding_window_support"] = "basic"  # or "advanced" in the future
        return info

    def _get_obj_class_shape(self):
        return Rectangle

    def _create_label(self, dto: PredictionBBox):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        geometry = Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(Tag(self._get_confidence_tag_meta(), dto.score))
        label = Label(geometry, obj_class, tags)
        return label

    def _get_custom_inference_settings(self) -> str:  # in yaml format
        settings = """confidence_threshold: 0.5"""
        return settings

    def predict(self, image_path: str, confidence_threshold: float) -> List[PredictionBBox]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_annotation(self, image_path: str, settings: Dict[str, Any]) -> Annotation:
        predictions = self.predict(image_path, settings)
        return self._predictions_to_annotation(image_path, predictions)

    @process_image_sliding_window
    @process_image_roi
    def inference_image_path(
        self,
        image_path: str,
        project_meta: ProjectMeta,
        state: Dict,
        settings: Dict = None,
    ):
        if settings is None:
            settings = self.get_inference_settings(state)
        logger.debug("Input path", extra={"path": image_path})
        ann = self.predict_annotation(
            image_path, settings=settings
        )
        if isinstance(ann, Annotation):
            return ann.to_json()
        else:
            return ann

    def visualize(self, predictions: List[PredictionBBox], image_path: str, vis_path: str):
        image = sly_image.read(image_path)
        ann = self._predictions_to_annotation(image_path, predictions)
        ann.draw_pretty(bitmap=image, output_path=vis_path, fill_rectangles=False)

    def serve(self):
        # import supervisely.nn.inference.instance_segmentation.dashboard.main_ui as main_ui
        # import supervisely.nn.inference.instance_segmentation.dashboard.deploy_ui as deploy_ui

        # @deploy_ui.deploy_btn.click
        # def deploy_model():
        # device = deploy_ui.device.get_value()
        # self.load_on_device(self._device)
        # print(f"✅ Model has been successfully loaded on {self._device.upper()} device")
        Progress("Deploying model ...", 1)
        super().serve()
        Progress("Model deployed", 1).iter_done_report()
