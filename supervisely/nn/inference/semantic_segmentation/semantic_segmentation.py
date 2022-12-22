from typing import Dict, List, Any
from supervisely.app.widgets.widget import Widget
from supervisely.geometry.bitmap import Bitmap
from supervisely.nn.prediction_dto import PredictionMask, PredictionSegmentation
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.sly_logger import logger
from supervisely.nn.inference.inference import Inference
from supervisely.task.progress import Progress
import numpy as np

class SemanticSegmentation(Inference):
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
        info["task type"] = "semantic segmentation"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return Bitmap

    def _create_label(self, dto: PredictionSegmentation):
        image_classes = np.unique(dto.mask)
        labels = []
        for class_idx in image_classes:
            class_mask = dto.mask == class_idx
            class_name = self.get_classes()[class_idx]
            obj_class = self.model_meta.get_obj_class(class_name)
            if obj_class is None:
                raise KeyError(
                    f"Class {class_name} not found in model classes {self.get_classes()}"
                )
            if not class_mask.any():  # skip empty masks
                logger.debug(f"Mask of class {class_name} is empty and will be sklipped")
                return None
            geometry = Bitmap(class_mask)
            label = Label(geometry, obj_class)
            labels.append(label)
        return labels

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionSegmentation]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionSegmentation]:
        raise NotImplementedError("Have to be implemented in child class If sliding_window_mode is 'advanced'.")

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
