import os
from re import L
from typing import List
import time
import yaml
from supervisely._utils import is_production, is_development, rand_str
from supervisely.app.fastapi.subapp import get_name_from_env
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
import supervisely.imaging.image as sly_image
import supervisely.io.fs as fs
from supervisely.sly_logger import logger

from supervisely.project.project_meta import ProjectMeta
from supervisely.app.fastapi.subapp import Application
from supervisely.api.api import Api

from pydantic import BaseModel


class ServeRequestBody(BaseModel):
    state: dict = {}
    context: dict = {}


class Inference:
    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir
        self._model_meta = None
        self._confidence = "confidence"
        self._app: Application = None
        self._api: Api = None
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
    def api(self) -> Api:
        if self._api is None:
            self._api = Api()
        return self._api

    def _get_obj_class_shape(self):
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            classes = []
            for name in self.get_classes():
                classes.append(ObjClass(name, self._get_obj_class_shape()))
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()  # @TODO: optimize, create if needed
        return self._model_meta

    def _get_confidence_tag_meta(self):
        tag_meta = self.model_meta.get_tag_meta(self._confidence)
        if tag_meta is None:
            tag_meta = TagMeta(self._confidence, TagValueType.ANY_NUMBER)
            self._model_meta = self._model_meta.add_tag_meta(tag_meta)
        return tag_meta

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

    def _get_custom_inference_settings() -> str:  # in yaml format
        return ""

    def _get_custom_inference_settings_dict(self) -> dict:
        return yaml.safe_load(self._get_custom_inference_settings())

    def inference_image_id(self, id: int) -> Annotation:
        image_info = self.api.image.get_info_by_id(id)
        image_path = os.path.join(__file__, rand_str(10), image_info.name)
        self.apiapi.image.download_path(id, image_path)
        ann = self.predict(image_path=image_path)
        fs.silent_remove(image_path.as_posix())
        return ann

    def validate_inference_settings(self, state: dict):
        settings = state.get("settings", {})

        for key, value in self._get_custom_inference_settings_dict().items():
            if key not in settings:
                logger.warn(
                    "Field {!r} not found in inference settings. Use default value {!r}".format(
                        key, value
                    )
                )

    @property
    def app(self) -> Application:
        return self._app

    def serve(self):
        # if is_production():
        #     print("prod")
        # else:
        #     print("localhostd")

        self._app = Application(headless=True)
        server = self._app.get_server()

        # @server.post("/get_session_info")
        # def get_session_info():
        #     print("start get_session_info")
        #     time.sleep(secs=25)
        #     print("finish get_session_info")
        #     return {}  # self.get_info()

        # @server.post("/get_custom_inference_settings")
        # def get_custom_inference_settings():
        #     print("start get_custom_inference_settings")
        #     time.sleep(secs=25)
        #     print("finish get_custom_inference_settings")
        #     # self._get_custom_inference_settings()
        #     return {"settings": ""}

        # @server.post("/get_output_classes_and_tags")
        # def get_output_classes_and_tags():
        #     print("start get_output_classes_and_tags")
        #     time.sleep(secs=25)
        #     print("finish get_output_classes_and_tags")
        #     return ProjectMeta().to_json()  # self.model_meta.to_json()

        # @server.post("/inference_image_id")
        # def inference_image_id(request_body: ServeRequestBody):
        #     return Annotation([256, 256]).to_json()

        # @server.post("/inference_image_id")
        # def inference_image_id(request_body: ServeRequestBody):
        # return Annotation([256, 256]).to_json()

        # state = request_body.state
        # logger.debug("Input data", extra={"state": state})
        # image_id = state["image_id"]

        # logger.debug("Input data", extra={"state": state})
        # image_id = state["image_id"]
        # ann = self.inference_image_id(image_id)
        # return ann.to_json()

        # @self._app.get_server().post("/get_session_info")
        # def get_session_info():
        #     return self.get_info()

        pass
