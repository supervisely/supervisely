from typing import Dict, List, Any
from supervisely.app.widgets.widget import Widget
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.prediction_dto import PredictionBBox
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.nn.inference.inference import Inference
from supervisely.task.progress import Progress


class ObjectDetection(Inference):
    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = "object detection"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
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

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionBBox]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionBBox]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def serve(self):
        if self.gui is not None:

            @self.gui.get_deploying_event()
            def load_model():
                device = self.gui.get_device()
                self.load_on_device(device)
                print(f"✅ Model has been successfully loaded on {self._device.upper()} device")

        Progress("Deploying model ...", 1)
        super().serve()
        Progress("Model deployed", 1).iter_done_report()
