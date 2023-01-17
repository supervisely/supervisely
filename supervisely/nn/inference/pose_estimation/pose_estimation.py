from typing import Dict, List, Any
from supervisely.app.widgets.widget import Widget
from supervisely.geometry.graph import GraphNodes
from supervisely.nn.prediction_dto import PredictionKeypoints
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.nn.inference.inference import Inference
from supervisely.task.progress import Progress
import supervisely as sly
from supervisely.geometry.graph import Node


class PoseEstimation(Inference):
    def get_info(self):
        info = super().get_info()
        info["task type"] = "pose estimation"
        # recommended parameters:
        # info["model_name"] = ""
        # info["checkpoint_name"] = ""
        # info["pretrained_on_dataset"] = ""
        # info["device"] = ""
        return info

    def _get_obj_class_shape(self):
        return GraphNodes

    def _create_label(self, dto: PredictionKeypoints):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        nodes = {}
        for i, keypoint in enumerate(dto.keypoints):
            x, y = keypoint
            nodes[str(i)] = Node(sly.PointLocation(y, x), disabled=False)
        geometry = GraphNodes(nodes)
        label = Label(geometry, obj_class)
        return label

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )
