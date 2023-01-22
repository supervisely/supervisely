from typing import Dict, List, Any, Union, Optional
from supervisely.app.widgets.widget import Widget
from supervisely.geometry.graph import GraphNodes
from supervisely.nn.prediction_dto import PredictionKeypoints
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.nn.inference.inference import Inference
from supervisely.task.progress import Progress
import supervisely as sly
from supervisely.geometry.graph import Node, KeypointsTemplate
from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.obj_class import ObjClass

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class PoseEstimation(Inference):
    def __init__(
        self,
        location: Optional[
            Union[str, List[str]]
        ] = None,  # folders of files with model or configs, from Team Files or external links
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        sliding_window_mode: Literal["basic", "advanced", "none"] = "basic",
        point_names: Optional[List[str]] = None,
        keypoints_template: Optional[KeypointsTemplate] = None,
    ):
        Inference.__init__(self, location, custom_inference_settings, sliding_window_mode)
        self.point_names = point_names
        self.keypoints_template = keypoints_template

    def _get_templates_dir(self):
        # template_dir = os.path.join(
        #     Path(__file__).parent.absolute(), "dashboard/templates"
        # )
        # return template_dir
        return None

    def _get_layout(self) -> Widget:
        # import supervisely.nn.inference.instance_segmentation.dashboard.main_ui as main_ui
        # return main_ui.menu
        return None

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

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            classes = []
            for name in self.get_classes():
                classes.append(
                    ObjClass(
                        name,
                        self._get_obj_class_shape(),
                        geometry_config=self.keypoints_template,
                    )
                )
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()
        return self._model_meta

    def _create_label(self, dto: PredictionKeypoints):
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if obj_class is None:
            raise KeyError(
                f"Class {dto.class_name} not found in model classes {self.get_classes()}"
            )
        nodes = []
        for label, coordinate in zip(dto.labels, dto.coordinates):
            x, y = coordinate
            nodes.append(Node(label=label, row=y, col=x))
        label = Label(GraphNodes(nodes), obj_class)
        return label

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionKeypoints]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

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
