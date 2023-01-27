import json
import os
from typing import List, Dict, Optional, Any, Union
from fastapi import Form, Response, UploadFile, status
from supervisely._utils import (
    is_debug_with_sly_net,
    rand_str,
    is_production,
)
from supervisely.app.fastapi.subapp import get_name_from_env
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_meta import TagMeta, TagValueType

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
import supervisely.imaging.image as sly_image
import supervisely.io.fs as fs
from supervisely.sly_logger import logger
import supervisely.io.env as env
import yaml

from supervisely.project.project_meta import ProjectMeta
from supervisely.app.fastapi.subapp import Application
from supervisely.app.content import StateJson, get_data_dir
from supervisely.app.fastapi.request import Request
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.nn.prediction_dto import Prediction
import supervisely.app.development as sly_app_development
from supervisely.imaging.color import get_predefined_colors
from supervisely.task.progress import Progress
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
)
import supervisely.nn.inference.gui as GUI

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Inference:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic",
        use_gui: Optional[bool] = False,
    ):
        if model_dir is None:
            model_dir = os.path.join(get_data_dir(), "models")
        self._model_dir = model_dir
        self._model_meta = None
        self._confidence = "confidence"
        self._app: Application = None
        self._api: Api = None
        self._task_id = None
        self._sliding_window_mode = sliding_window_mode
        if custom_inference_settings is None:
            custom_inference_settings = {}
        if isinstance(custom_inference_settings, str):
            if fs.file_exists(custom_inference_settings):
                with open(custom_inference_settings, "r") as f:
                    custom_inference_settings = f.read()
            else:
                raise FileNotFoundError(f"{custom_inference_settings} file not found.")
        self._custom_inference_settings = custom_inference_settings
        self._use_gui = use_gui
        self._gui = None

    def _prepare_device(self, device):
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception as e:
                logger.warn(
                    f"Device auto detection failed, set to default 'cpu', reason: {repr(e)}"
                )
                device = "cpu"

    def get_ui(self) -> Widget:
        if not self._use_gui:
            return None
        return self.gui.get_ui()

    def get_ui_class(self) -> GUI.BaseInferenceGUI:
        return GUI.InferenceGUI

    def get_models(self) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        raise RuntimeError("Have to be implemented in child class after inheritance")

    def download(self, src_path: str, dst_path: str = None):
        if dst_path is None:
            dst_path = os.path.join(self._model_dir, basename)
        if fs.is_on_agent(src_path) or is_production():
            team_id = env.team_id()
            basename = os.path.basename(os.path.normpath(src_path))
            progress = Progress(f"Downloading {basename}...", 1, is_size=True, need_info_log=True)
            if fs.dir_exists(src_path) or fs.file_exists(
                src_path
            ):  # only during debug, has no effect in production
                dst_path = os.path.abspath(src_path)
                logger.info(f"File {dst_path} found.")
            elif self._api.file.dir_exists(team_id, src_path) and src_path.endswith(
                "/"
            ):  # folder from Team Files
                logger.info(f"Remote directory in Team Files: {src_path}")
                logger.info(f"Local directory: {dst_path}")
                sizeb = self._api.file.get_directory_size(team_id, src_path)
                progress.set(current=0, total=sizeb)
                self._api.file.download_directory(
                    team_id,
                    src_path,
                    dst_path,
                    progress.iters_done_report,
                )
                logger.info(
                    f"📥 Directory {basename} has been successfully downloaded from Team Files"
                )
                logger.info(f"Directory {basename} path: {dst_path}")
            elif self._api.file.exists(team_id, src_path):  # file from Team Files
                file_info = self._api.file.get_info_by_path(env.team_id(), src_path)
                progress.set(current=0, total=file_info.sizeb)
                self._api.file.download(
                    team_id, src_path, dst_path, progress_cb=progress.iters_done_report
                )
                logger.info(f"📥 File {basename} has been successfully downloaded from Team Files")
                logger.info(f"File {basename} path: {dst_path}")
            else:  # external url
                fs.download(src_path, dst_path, progress=progress)
                logger.info(f"📥 File {basename} has been successfully downloaded.")
                logger.info(f"File {basename} path: {dst_path}")
        else:
            dst_path = os.path.abspath(src_path)
            logger.info(f"File {dst_path} found.")
        return dst_path

    def _preprocess_models_list(self, models_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # fill skipped columns
        all_columns = []
        for model_dict in models_list:
            cols = model_dict.keys()
            all_columns.extend([col for col in cols if col not in all_columns])
        for i, model_dict in enumerate(models_list):
            for col in all_columns:
                if col not in model_dict.keys():
                    models_list[i][col] = "-"
        return models_list

    def load_on_device(
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def get_classes(self) -> List[str]:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def get_info(self) -> dict:
        return {
            "app_name": get_name_from_env(default="Neural Network Serving"),
            "session_id": self.task_id,
            "number_of_classes": len(self.get_classes()),
            "sliding_window_support": self.sliding_window_mode,
            "videos_support": True,
        }

    @property
    def sliding_window_mode(self) -> Literal["basic", "advanced", "none"]:
        return self._sliding_window_mode

    @property
    def api(self) -> Api:
        if self._api is None:
            self._api = Api()
        return self._api

    @property
    def gui(self) -> GUI.BaseInferenceGUI:
        return self._gui

    def _get_obj_class_shape(self):
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            colors = get_predefined_colors(len(self.get_classes()))
            classes = []
            for name, rgb in zip(self.get_classes(), colors):
                classes.append(ObjClass(name, self._get_obj_class_shape(), rgb))
            self._model_meta = ProjectMeta(classes)
            self._get_confidence_tag_meta()
        return self._model_meta

    @property
    def task_id(self) -> int:
        return self._task_id

    def _get_confidence_tag_meta(self):
        tag_meta = self.model_meta.get_tag_meta(self._confidence)
        if tag_meta is None:
            tag_meta = TagMeta(self._confidence, TagValueType.ANY_NUMBER)
            self._model_meta = self._model_meta.add_tag_meta(tag_meta)
        return tag_meta

    def _create_label(self, dto: Prediction) -> Label:
        raise NotImplementedError("Have to be implemented in child class")

    def _predictions_to_annotation(
        self, image_path: str, predictions: List[Prediction]
    ) -> Annotation:
        labels = []
        for prediction in predictions:
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            if isinstance(label, list):
                labels.extend(label)
                continue
            labels.append(label)

        # create annotation with correct image resolution
        ann = Annotation.from_img_path(image_path)
        ann = ann.add_labels(labels)
        return ann

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def custom_inference_settings(self) -> Union[Dict[str, any], str]:
        return self._custom_inference_settings

    @property
    def custom_inference_settings_dict(self) -> Dict[str, any]:
        if isinstance(self._custom_inference_settings, dict):
            return self._custom_inference_settings
        else:
            return yaml.safe_load(self._custom_inference_settings)

    @process_image_sliding_window
    @process_image_roi
    def _inference_image_path(
        self,
        image_path: str,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ):
        inference_mode = settings.get("inference_mode", "full_image")
        logger.debug(
            "Inferring image_path:", extra={"inference_mode": inference_mode, "path": image_path}
        )

        if inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced":
            predictions = self.predict_raw(image_path=image_path, settings=settings)
        else:
            predictions = self.predict(image_path=image_path, settings=settings)
        ann = self._predictions_to_annotation(image_path, predictions)

        logger.debug(
            f"Inferring image_path done. pred_annotation:",
            extra=dict(w=ann.img_size[1], h=ann.img_size[0], n_labels=len(ann.labels)),
        )
        return ann

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def _get_inference_settings(self, state: dict):
        settings = state.get("settings", {})
        if settings is None:
            settings = {}
        if "rectangle" in state.keys():
            settings["rectangle"] = state["rectangle"]
        settings["sliding_window_mode"] = self.sliding_window_mode

        for key, value in self.custom_inference_settings_dict.items():
            if key not in settings:
                logger.warn(
                    f"Field {key} not found in inference settings. Use default value {value}"
                )
                settings[key] = value
        return settings

    @property
    def app(self) -> Application:
        return self._app

    def visualize(
        self,
        predictions: List[Prediction],
        image_path: str,
        vis_path: str,
        thickness: Optional[int] = None,
    ):
        image = sly_image.read(image_path)
        ann = self._predictions_to_annotation(image_path, predictions)
        ann.draw_pretty(
            bitmap=image, thickness=thickness, output_path=vis_path, fill_rectangles=False
        )

    def _inference_image(self, state: dict, file: UploadFile):
        logger.debug("Inferring image...", extra={"state": state})
        settings = self._get_inference_settings(state)
        image_path = os.path.join(get_data_dir(), f"{rand_str(10)}_{file.filename}")
        image_np = sly_image.read_bytes(file.file.read())
        logger.debug("Inference settings:", extra=settings)
        logger.debug("Image info:", extra={"w": image_np.shape[1], "h": image_np.shape[0]})
        sly_image.write(image_path, image_np)
        data_to_return = {}
        ann = self._inference_image_path(
            image_path=image_path,
            settings=settings,
            data_to_return=data_to_return,
        )
        fs.silent_remove(image_path)
        return {"annotation": ann.to_json(), "data": data_to_return}

    def _inference_batch(self, state: dict, files: List[UploadFile]):
        logger.debug("Inferring image batch...", extra={"state": state})
        paths = []
        temp_dir = os.path.join(get_data_dir(), rand_str(10))
        fs.mkdir(temp_dir)
        for file in files:
            image_path = os.path.join(temp_dir, f"{rand_str(10)}_{file.filename}")
            image_np = sly_image.read_bytes(file.file.read())
            sly_image.write(image_path, image_np)
            paths.append(image_path)
        results = self._inference_images_dir(paths, state)
        fs.remove_dir(temp_dir)
        return results

    def _inference_batch_ids(self, api: Api, state: dict):
        logger.debug("Inferring image_ids batch...", extra={"state": state})
        ids = state["batch_ids"]
        infos = api.image.get_info_by_id_batch(ids)
        paths = []
        temp_dir = os.path.join(get_data_dir(), rand_str(10))
        fs.mkdir(temp_dir)
        for info in infos:
            paths.append(os.path.join(temp_dir, f"{rand_str(10)}_{info.name}"))
        api.image.download_paths(
            infos[0].dataset_id, ids, paths
        )  # TODO: check if this is correct (from the same ds)
        results = self._inference_images_dir(paths, state)
        fs.remove_dir(temp_dir)
        return results

    def _inference_images_dir(self, img_paths: List[str], state: Dict):
        logger.debug("Inferring images_dir (or batch)...")
        settings = self._get_inference_settings(state)
        logger.debug("Inference settings:", extra=settings)
        n_imgs = len(img_paths)
        results = []
        for i, image_path in enumerate(img_paths):
            data_to_return = {}
            logger.debug(f"Inferring image {i+1}/{n_imgs}.", extra={"path": image_path})
            ann = self._inference_image_path(
                image_path=image_path,
                settings=settings,
                data_to_return=data_to_return,
            )
            results.append({"annotation": ann.to_json(), "data": data_to_return})
        return results

    def _inference_image_id(self, api: Api, state: dict):
        logger.debug("Inferring image_id...", extra={"state": state})
        settings = self._get_inference_settings(state)
        image_id = state["image_id"]
        image_info = api.image.get_info_by_id(image_id)
        image_path = os.path.join(get_data_dir(), f"{rand_str(10)}_{image_info.name}")
        api.image.download_path(image_id, image_path)
        logger.debug("Inference settings:", extra=settings)
        logger.debug(
            "Image info:", extra={"id": image_id, "w": image_info.width, "h": image_info.height}
        )
        logger.debug(f"Downloaded path: {image_path}")
        data_to_return = {}
        ann = self._inference_image_path(
            image_path=image_path,
            settings=settings,
            data_to_return=data_to_return,
        )
        fs.silent_remove(image_path)
        return {"annotation": ann.to_json(), "data": data_to_return}

    def _inference_image_url(self, api: Api, state: dict):
        logger.debug("Inferring image_url...", extra={"state": state})
        settings = self._get_inference_settings(state)
        image_url = state["image_url"]
        ext = fs.get_file_ext(image_url)
        if ext == "":
            ext = ".jpg"
        image_path = os.path.join(get_data_dir(), rand_str(15) + ext)
        fs.download(image_url, image_path)
        logger.debug("Inference settings:", extra=settings)
        logger.debug(f"Downloaded path: {image_path}")
        data_to_return = {}
        ann = self._inference_image_path(
            image_path=image_path,
            settings=settings,
            data_to_return=data_to_return,
        )
        fs.silent_remove(image_path)
        return {"annotation": ann.to_json(), "data": data_to_return}

    def _inference_video_id(self, api: Api, state: dict):
        from supervisely.nn.inference.video_inference import InferenceVideoInterface

        logger.debug("Inferring video_id...", extra={"state": state})
        video_info = api.video.get_info_by_id(state["videoId"])
        logger.debug(
            f"Video info:",
            extra=dict(
                w=video_info.frame_width,
                h=video_info.frame_height,
                n_frames=video_info.frames_count,
            ),
        )

        video_images_path = os.path.join(get_data_dir(), rand_str(15))
        inf_video_interface = InferenceVideoInterface(
            api=api,
            start_frame_index=state.get("startFrameIndex", 0),
            frames_count=state.get("framesCount", video_info.frames_count - 1),
            frames_direction=state.get("framesDirection", "forward"),
            video_info=video_info,
            imgs_dir=video_images_path,
        )
        inf_video_interface.download_frames()

        settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=settings)

        n_frames = len(inf_video_interface.images_paths)
        logger.debug(f"Total frames to infer: {n_frames}")

        results = []
        for i, image_path in enumerate(inf_video_interface.images_paths):
            logger.debug(f"Inferring frame {i+1}/{n_frames}:", extra={"image_path": image_path})
            data_to_return = {}
            ann = self._inference_image_path(
                image_path=image_path,
                settings=settings,
                data_to_return=data_to_return,
            )
            results.append({"annotation": ann.to_json(), "data": data_to_return})
            logger.debug(f"Frame {i+1} done.")
        fs.remove_dir(video_images_path)
        return results

    def serve(self):
        Progress("Deploying model ...", 1)
        if self._use_gui:
            models = self.get_models()
            if isinstance(models, list):
                models = self._preprocess_models_list(models)
            elif isinstance(models, dict):
                for model_group in models.keys():
                    models[model_group]["checkpoints"] = self._preprocess_models_list(
                        models[model_group]["checkpoints"]
                    )
            self._gui = self.get_ui_class()(models)

            @self.gui.serve_button.click
            def load_model():
                # TODO: maybe add custom logic?
                device = self.gui.get_device()
                self.load_on_device(self._model_dir, device)
                self.gui.set_deployed()

        if is_debug_with_sly_net():
            # advanced debug for Supervisely Team
            logger.warn(
                "Serving is running in advanced development mode with Supervisely VPN Network"
            )
            team_id = env.team_id()
            # sly_app_development.supervisely_vpn_network(action="down") # for debug
            sly_app_development.supervisely_vpn_network(action="up")
            task = sly_app_development.create_debug_task(team_id, port="8000")
            self._task_id = task["id"]
        else:
            self._task_id = env.task_id()

        self._app = Application(layout=self.get_ui())
        server = self._app.get_server()

        if not self._use_gui:
            Progress("Model deployed", 1).iter_done_report()

        @server.post(f"/get_session_info")
        def get_session_info():
            return self.get_info()

        @server.post("/get_custom_inference_settings")
        def get_custom_inference_settings():
            return {"settings": self.custom_inference_settings}

        @server.post("/get_output_classes_and_tags")
        def get_output_classes_and_tags():
            return self.model_meta.to_json()

        @server.post("/inference_image_id")
        def inference_image_id(request: Request):
            return self._inference_image_id(request.api, request.state)

        @server.post("/inference_image_url")
        def inference_image_url(request: Request):
            return self._inference_image_url(request.api, request.state)

        @server.post("/inference_batch_ids")
        def inference_batch_ids(request: Request):
            return self._inference_batch_ids(request.api, request.state)

        @server.post("/inference_video_id")
        def inference_video_id(request: Request):
            return {"ann": self._inference_video_id(request.api, request.state)}

        @server.post("/inference_image")
        def inference_image(
            response: Response, files: List[UploadFile], settings: str = Form("{}")
        ):
            if len(files) != 1:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"Only one file expected but got {len(files)}"
            try:
                state = json.loads(settings)
                if type(state) != dict:
                    response.status_code = status.HTTP_400_BAD_REQUEST
                    return "Settings is not json object"
                return self._inference_image(state, files[0])
            except (json.decoder.JSONDecodeError, TypeError) as e:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"Cannot decode settings: {e}"
            except sly_image.UnsupportedImageFormat:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"File has unsupported format. Supported formats: {sly_image.SUPPORTED_IMG_EXTS}"

        @server.post("/inference_batch")
        def inference_batch(
            response: Response, files: List[UploadFile], settings: str = Form("{}")
        ):
            try:
                state = json.loads(settings)
                if type(state) != dict:
                    response.status_code = status.HTTP_400_BAD_REQUEST
                    return "Settings is not json object"
                return self._inference_batch(state, files)
            except (json.decoder.JSONDecodeError, TypeError) as e:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"Cannot decode settings: {e}"
            except sly_image.UnsupportedImageFormat:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"File has unsupported format. Supported formats: {sly_image.SUPPORTED_IMG_EXTS}"
