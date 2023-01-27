import json
import os
import uuid
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Union
from fastapi import Form, Response, UploadFile, status
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
import supervisely.io.env as env
import yaml

from supervisely.project.project_meta import ProjectMeta
from supervisely.app.fastapi.subapp import Application
from supervisely.app.content import StateJson, get_data_dir
from supervisely.app.fastapi.request import Request
from supervisely.api.api import Api
from supervisely.nn.prediction_dto import Prediction
import supervisely.app.development as sly_app_development
from supervisely.imaging.color import get_predefined_colors
from supervisely.task.progress import Progress
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
)

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Inference:
    def __init__(
        self,
        location: Optional[
            Union[str, List[str]]
        ] = None,  # folders of files with model or configs, from Team Files or external links
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        sliding_window_mode: Literal["basic", "advanced", "none"] = "basic",
    ):
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
        self._headless = True
        self._inference_requests = {}
        self._executor = ThreadPoolExecutor()

        self._prepare_model_files(location)

    def _download_from_location(self, location_link, local_files_path):
        if fs.is_on_agent(location_link) or is_production():
            team_id = env.team_id()
            basename = os.path.basename(os.path.normpath(location_link))
            local_path = os.path.join(local_files_path, basename)
            progress = Progress(f"Downloading {basename}...", 1, is_size=True, need_info_log=True)
            if fs.dir_exists(location_link) or fs.file_exists(location_link):
                # only during debug, has no effect in production
                local_path = os.path.abspath(location_link)
            elif self.api.file.dir_exists(team_id, location_link) and location_link.endswith(
                "/"
            ):  # folder from Team Files
                logger.info(f"Remote directory in Team Files: {location_link}")
                logger.info(f"Local directory: {local_path}")
                sizeb = self.api.file.get_directory_size(team_id, location_link)
                progress.set(current=0, total=sizeb)
                self.api.file.download_directory(
                    team_id,
                    location_link,
                    local_path,
                    progress.iters_done_report,
                )
                print(f"📥 Directory {basename} has been successfully downloaded from Team Files")
            elif self.api.file.exists(team_id, location_link):  # file from Team Files
                file_info = self.api.file.get_info_by_path(env.team_id(), location_link)
                progress.set(current=0, total=file_info.sizeb)
                self.api.file.download(
                    team_id, location_link, local_path, progress_cb=progress.iters_done_report
                )
                print(f"📥 File {basename} has been successfully downloaded from Team Files")
            else:  # external url
                fs.download(location_link, local_path, progress=progress)
                print(f"📥 File {basename} has been successfully downloaded.")
            print(f"File {basename} path: {local_path}")
        else:
            local_path = os.path.abspath(location_link)
        return local_path

    def _prepare_model_files(self, location: Optional[Union[str, List[str]]] = None):
        self._location = None
        if location is None:
            return
        local_files_path = os.path.join(get_data_dir(), "model")
        fs.mkdir(local_files_path)
        if isinstance(location, str):
            self._location = self._download_from_location(location, local_files_path)
        else:
            self._location = []
            for location_link in location:
                self._location.append(self._download_from_location(location_link, local_files_path))

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

    def _get_templates_dir(self):
        return None
        # raise NotImplementedError("Have to be implemented in child class")

    def _get_layout(self):
        return None
        # raise NotImplementedError("Have to be implemented in child class")

    def load_on_device(
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu"
    ):
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def get_classes(self) -> List[str]:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def get_info(self) -> dict:
        return {
            "app_name": get_name_from_env(default="Neural Network Serving"),
            "session_id": self.task_id,
            "model_files": self.location,
            "number_of_classes": len(self.get_classes()),
            "sliding_window_support": self.sliding_window_mode,
            "videos_support": True,
            "async_video_inference_support": True,
        }

    @property
    def location(self) -> Union[str, List[str]]:
        return self._location

    @property
    def sliding_window_mode(self) -> Literal["basic", "advanced", "none"]:
        return self._sliding_window_mode

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
            ann = self._predictions_to_annotation(image_path, predictions)
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
        logger.debug("Inferring batch...", extra={"state": state})
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
        logger.debug("Inferring batch_ids...", extra={"state": state})
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

    def _inference_video_id(self, api: Api, state: dict, async_inference_request_uuid: str = None):
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

        if async_inference_request_uuid is not None:
            try:
                inference_request = self._inference_requests[async_inference_request_uuid]
            except Exception as ex:
                import traceback

                logger.error(traceback.format_exc())
                raise RuntimeError(
                    f"async_inference_request_uuid {async_inference_request_uuid} was given, "
                    f"but there is no such uuid in 'self._inference_requests' ({len(self._inference_requests)} items)"
                )
            sly_progress: Progress = inference_request["progress"]
            sly_progress.total = n_frames

        results = []
        for i, image_path in enumerate(inf_video_interface.images_paths):
            if (
                async_inference_request_uuid is not None
                and inference_request["cancel_inference"] is True
            ):
                logger.debug(
                    f"Cancelling inference video...",
                    extra={"inference_request_uuid": async_inference_request_uuid},
                )
                results = []
                break
            logger.debug(f"Inferring frame {i+1}/{n_frames}:", extra={"image_path": image_path})
            data_to_return = {}
            ann = self._inference_image_path(
                image_path=image_path,
                settings=settings,
                data_to_return=data_to_return,
            )
            if async_inference_request_uuid is not None:
                sly_progress.iter_done()
            results.append({"annotation": ann.to_json(), "data": data_to_return})
            logger.debug(f"Frame {i+1} done.")

        fs.remove_dir(video_images_path)
        if async_inference_request_uuid is not None and len(results) > 0:
            inference_request["result"] = {"ann": results}
        return results

    def _on_inference_start(self, inference_request_uuid):
        inference_request = {
            "progress": Progress("Inferring model...", total_cnt=1),
            "is_inferring": True,
            "cancel_inference": False,
            "result": None,
        }
        self._inference_requests[inference_request_uuid] = inference_request

    def _on_inference_end(self, future, inference_request_uuid):
        logger.debug("_on_inference_end() callback")
        self._inference_requests[inference_request_uuid]["is_inferring"] = False

    def serve(self):
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

        # headless=self._headless,
        self._app = Application(layout=self._get_layout(), templates_dir=self._get_templates_dir())
        server = self._app.get_server()

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
            # TODO: уточнить ждет ли применялка лист или дикт тоже понимает
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

        @server.post("/inference_video_id_async")
        def inference_video_id_async(request: Request):
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._inference_video_id, request.api, request.state, inference_request_uuid
            )
            end_callback = partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            logger.debug(
                "Inference has scheduled from 'inference_video_id_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post(f"/get_inference_progress")
        def get_inference_progress(request: Request):
            inference_request_uuid = request.state.get("inference_request_uuid")
            if not inference_request_uuid:
                return {"message": "Error: 'inference_request_uuid' is required."}
            inference_request = self._inference_requests[inference_request_uuid].copy()
            sly_progress: Progress = inference_request["progress"]
            inference_request["progress"] = {
                "current": sly_progress.current,
                "total": sly_progress.total,
            }
            logger.debug(
                f"Sending inference progress with uuid {inference_request_uuid}:",
                extra=inference_request,
            )
            return inference_request

        @server.post(f"/stop_inference")
        def stop_inference(request: Request):
            inference_request_uuid = request.state.get("inference_request_uuid")
            if not inference_request_uuid:
                return {"message": "Error: 'inference_request_uuid' is required.", "success": False}
            inference_request = self._inference_requests[inference_request_uuid]
            inference_request["cancel_inference"] = True
            return {"message": "Inference will be stopped.", "success": True}
