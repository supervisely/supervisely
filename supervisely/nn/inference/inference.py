import os
from typing import List, Dict
import yaml
from supervisely._utils import (
    is_production,
    is_development,
    is_debug_with_sly_net,
    rand_str,
)
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
from supervisely.app.content import StateJson, get_data_dir
from supervisely.app.fastapi.request import Request
from supervisely.api.api import Api
import supervisely.app.development as sly_app_development
from distinctipy import distinctipy

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Inference:
    def __init__(
        self,
        model_dir: str = None,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self._model_dir = model_dir
        self._device = device
        self._model_meta = None
        self._confidence = "confidence"
        self._app: Application = None
        self._api: Api = None

        self._headless = False
        # self._template_dir = None
        # self._template_dir = Path(__file__).parent.absolute()
        if is_production():
            if os.environ.get("_SPAWN_USER_ID") is None:
                logger.debug("Running serving on localhost with enabled UI")
            else:
                logger.debug(
                    "Running serving on Supervisely platform in production mode"
                )
                raise NotImplementedError("TBD - download directory")
        elif is_development():
            self._headless = True
            pass

    def _get_templates_dir(self):
        raise NotImplementedError("Have to be implemented in child class")

    def load_on_device(
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu"
    ):
        raise NotImplementedError(
            "Have to be implemented in child class after inheritance"
        )

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
            colors = distinctipy.get_colors(len(self.get_classes()))
            classes = []
            for name, color in zip(self.get_classes(), colors):
                rgb = distinctipy.get_rgb256(color)
                classes.append(ObjClass(name, self._get_obj_class_shape(), rgb))
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()
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
        raise NotImplementedError("Have to be implemented in child class")

    def visualize(self, predictions: List, image_path: str, vis_path: str):
        raise NotImplementedError("Have to be implemented in child class")

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

    def inference_image_path(
        self, image_path, project_meta: ProjectMeta, state: Dict, settings: Dict = None
    ):
        raise NotImplementedError()

    def _get_custom_inference_settings_dict(self) -> dict:
        return yaml.safe_load(self._get_custom_inference_settings())

    def get_inference_settings(self, state: dict):
        settings = state.get("settings", {})
        for key, value in self._get_custom_inference_settings_dict().items():
            if key not in settings:
                logger.warn(
                    f"Field {key} not found in inference settings. Use default value {value}"
                )
                settings[key] = value
        return settings

    @property
    def app(self) -> Application:
        return self._app

    def inference_batch_ids(self, api: Api, state: dict):
        ids = state["batch_ids"]
        infos = api.image.get_info_by_id_batch(ids)
        paths = []
        temp_dir = os.path.join(get_data_dir(), rand_str(10))
        fs.mkdir(temp_dir)
        for info in infos:
            paths.append(os.path.join(temp_dir, f"{rand_str(10)}_{info.name}"))
        api.image.download_paths(infos[0].dataset_id, ids, paths)
        results = self.inference_images_dir(paths, state)
        fs.remove_dir(temp_dir)
        return results

    def inference_images_dir(self, img_paths: List[str], state: Dict):
        settings = self.get_inference_settings(state)
        annotations = []
        for image_path in img_paths:
            ann_json = self.inference_image_path(
                image_path=image_path,
                project_meta=self.model_meta,
                state=state,
                settings=settings,
            )
            annotations.append(ann_json)
        return annotations

    def inference_image_id(self, api: Api, state: dict):
        logger.debug("Input state", extra={"state": state})
        settings = self.get_inference_settings(state)
        image_id = state["image_id"]
        image_info = api.image.get_info_by_id(image_id)
        image_path = os.path.join(get_data_dir(), f"{rand_str(10)}_{image_info.name}")
        api.image.download_path(image_id, image_path)
        ann_json = self.inference_image_path(
            image_path=image_path,
            project_meta=self.model_meta,
            state=state,
            settings=settings,
        )
        fs.silent_remove(image_path)
        return ann_json

    def inference_image_url(self, api: Api, state: dict):
        logger.debug("Input data", extra={"state": state})
        settings = self.get_inference_settings(state)
        image_url = state["image_url"]
        ext = fs.get_file_ext(image_url)
        if ext == "":
            ext = ".jpg"
        image_path = os.path.join(get_data_dir(), rand_str(15) + ext)
        fs.download(image_url, image_path)
        ann_json = self.inference_image_path(
            image_path=image_path,
            project_meta=self.model_meta,
            state=state,
            settings=settings,
        )
        fs.silent_remove(image_path)
        return ann_json

    def serve(self):
        if is_debug_with_sly_net():
            # advanced debug for Supervisely Team
            logger.warn(
                "Serving is running in advanced development mode with Supervisely VPN Network"
            )
            team_id = int(os.environ["context.teamId"])
            # sly_app_development.supervisely_vpn_network(action="down") # for debug
            sly_app_development.supervisely_vpn_network(action="up")
            task = sly_app_development.create_debug_task(team_id, port="8000")

        # headless=self._headless,
        self._app = Application(templates_dir=self._get_templates_dir())
        server = self._app.get_server()

        @server.post(f"/get_session_info")
        def get_session_info():
            return self.get_info()

        @server.post("/get_custom_inference_settings")
        def get_custom_inference_settings():
            settings = self._get_custom_inference_settings()
            return {"settings": settings}

        @server.post("/get_output_classes_and_tags")
        def get_output_classes_and_tags():
            return self.model_meta.to_json()

        @server.post("/inference_image_id")
        def inference_image_id(request: Request):
            return self.inference_image_id(request.api, request.state)

        @server.post("/inference_image_url")
        def inference_image_url(request: Request):
            return self.inference_image_url(request.api, request.state)

        @server.post("/inference_batch_ids")
        def inference_batch_ids(request: Request):
            return self.inference_batch_ids(request.api, request.state)

        @server.post("/inference_video_id")
        def inference_video_id(request: Request):
            raise NotImplementedError()
