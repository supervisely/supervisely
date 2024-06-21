import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import requests
import yaml
from fastapi import Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from requests.structures import CaseInsensitiveDict

import supervisely.app.development as sly_app_development
import supervisely.imaging.image as sly_image
import supervisely.io.env as env
import supervisely.io.fs as fs
import supervisely.nn.inference.gui as GUI
from supervisely import DatasetInfo, ProjectInfo, VideoAnnotation, batched
from supervisely._utils import (
    add_callback,
    is_debug_with_sly_net,
    is_production,
    rand_str,
)
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.api import Api
from supervisely.app.content import StateJson, get_data_dir
from supervisely.app.exceptions import DialogWindowError
from supervisely.app.fastapi.subapp import (
    Application,
    call_on_autostart,
    get_name_from_env,
)
from supervisely.app.widgets import Card, Container, Widget
from supervisely.app.widgets.editor.editor import Editor
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
    process_images_batch_roi,
    process_images_batch_sliding_window,
)
from supervisely.imaging.color import get_predefined_colors
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.prediction_dto import Prediction
from supervisely.project import ProjectType
from supervisely.project.download import download_to_cache, read_from_cached_project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress

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
        multithread_inference: Optional[bool] = True,
    ):
        if model_dir is None:
            model_dir = os.path.join(get_data_dir(), "models")
            fs.mkdir(model_dir)
        self._model_dir = model_dir
        self._model_served = False
        self._model_meta = None
        self._confidence = "confidence"
        self._app: Application = None
        self._api: Api = None
        self._task_id = None
        self._sliding_window_mode = sliding_window_mode
        self._autostart_delay_time = 5 * 60  # 5 min
        self._tracker = None
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

        self.load_on_device = LOAD_ON_DEVICE_DECORATOR(self.load_on_device)
        self.load_on_device = add_callback(self.load_on_device, self._set_served_callback)

        self.load_model = LOAD_MODEL_DECORATOR(self.load_model)
        self.load_model = add_callback(self.load_model, self._set_served_callback)

        if use_gui:
            initialize_custom_gui_method = getattr(self, "initialize_custom_gui", None)
            original_initialize_custom_gui_method = getattr(
                Inference, "initialize_custom_gui", None
            )
            if initialize_custom_gui_method.__func__ is not original_initialize_custom_gui_method:
                self._gui = GUI.ServingGUI()
                self._user_layout = self.initialize_custom_gui()
            else:
                self.initialize_gui()

            def on_serve_callback(gui: Union[GUI.InferenceGUI, GUI.ServingGUI]):
                Progress("Deploying model ...", 1)

                if isinstance(self.gui, GUI.ServingGUI):
                    deploy_params = self.get_params_from_gui()
                    self.load_model(**deploy_params)
                    self.update_gui(self._model_served)
                else:  # GUI.InferenceGUI
                    device = gui.get_device()
                    self.load_on_device(self._model_dir, device)
                gui.show_deployed_model_info(self)

            def on_change_model_callback(gui: Union[GUI.InferenceGUI, GUI.ServingGUI]):
                self.shutdown_model()
                if isinstance(self.gui, GUI.ServingGUI):
                    self._api_request_model_layout.unlock()
                    self._api_request_model_layout.hide()
                    self.update_gui(self._model_served)
                    self._user_layout_card.show()

            self.gui.on_change_model_callbacks.append(on_change_model_callback)
            self.gui.on_serve_callbacks.append(on_serve_callback)

        self._inference_requests = {}
        max_workers = 1 if not multithread_inference else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.predict = self._check_serve_before_call(self.predict)
        self.predict_raw = self._check_serve_before_call(self.predict_raw)
        self.get_info = self._check_serve_before_call(self.get_info)

        self.cache = InferenceImageCache(
            maxsize=env.smart_cache_size(),
            ttl=env.smart_cache_ttl(),
            is_persistent=True,
            base_folder=env.smart_cache_container_dir(),
        )

    def _prepare_device(self, device):
        if device is None:
            try:
                import torch  # pylint: disable=import-error

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

    def initialize_custom_gui(self) -> Widget:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def update_gui(self, is_model_deployed: bool = True) -> None:
        if isinstance(self.gui, GUI.ServingGUI):
            if is_model_deployed:
                self._user_layout_card.lock()
            else:
                self._user_layout_card.unlock()

    def set_params_to_gui(self, deploy_params: dict) -> None:
        if isinstance(self.gui, GUI.ServingGUI):
            self._user_layout_card.hide()
            self._api_request_model_info.set_text(json.dumps(deploy_params), "json")
            self._api_request_model_layout.show()

    def get_params_from_gui(self) -> dict:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def initialize_gui(self) -> None:
        models = self.get_models()
        support_pretrained_models = True
        if isinstance(models, list):
            if len(models) > 0:
                models = self._preprocess_models_list(models)
            else:
                support_pretrained_models = False
        elif isinstance(models, dict):
            for model_group in models.keys():
                models[model_group]["checkpoints"] = self._preprocess_models_list(
                    models[model_group]["checkpoints"]
                )
        self._gui = GUI.InferenceGUI(
            models,
            self.api,
            support_pretrained_models=support_pretrained_models,
            support_custom_models=self.support_custom_models(),
            add_content_to_pretrained_tab=self.add_content_to_pretrained_tab,
            add_content_to_custom_tab=self.add_content_to_custom_tab,
            custom_model_link_type=self.get_custom_model_link_type(),
        )

    def support_custom_models(self) -> bool:
        return True

    def add_content_to_pretrained_tab(self, gui: GUI.BaseInferenceGUI) -> Widget:
        return None

    def add_content_to_custom_tab(self, gui: GUI.BaseInferenceGUI) -> Widget:
        return None

    def get_custom_model_link_type(self) -> Literal["file", "folder"]:
        return "file"

    def get_models(
        self,
    ) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        return []

    def download(self, src_path: str, dst_path: str = None):
        basename = os.path.basename(os.path.normpath(src_path))
        if dst_path is None:
            dst_path = os.path.join(self._model_dir, basename)
        if self.gui is not None:
            progress = self.gui.download_progress
        else:
            progress = None

        if fs.dir_exists(src_path) or fs.file_exists(
            src_path
        ):  # only during debug, has no effect in production
            dst_path = os.path.abspath(src_path)
            logger.info(f"File {dst_path} found.")
        elif src_path.startswith("/"):  # folder from Team Files
            team_id = env.team_id()

            if src_path.endswith("/") and self.api.file.dir_exists(team_id, src_path):

                def download_dir(team_id, src_path, dst_path, progress_cb=None):
                    self.api.file.download_directory(
                        team_id,
                        src_path,
                        dst_path,
                        progress_cb=progress_cb,
                    )

                logger.info(f"Remote directory in Team Files: {src_path}")
                logger.info(f"Local directory: {dst_path}")
                sizeb = self.api.file.get_directory_size(team_id, src_path)

                if progress is None:
                    download_dir(team_id, src_path, dst_path)
                else:
                    self.gui.download_progress.show()
                    with progress(
                        message="Downloading directory from Team Files...",
                        total=sizeb,
                        unit="bytes",
                        unit_scale=True,
                    ) as pbar:
                        download_dir(team_id, src_path, dst_path, pbar.update)
                logger.info(
                    f"📥 Directory {basename} has been successfully downloaded from Team Files"
                )
                logger.info(f"Directory {basename} path: {dst_path}")
            elif self.api.file.exists(team_id, src_path):  # file from Team Files

                def download_file(team_id, src_path, dst_path, progress_cb=None):
                    self.api.file.download(team_id, src_path, dst_path, progress_cb=progress_cb)

                file_info = self.api.file.get_info_by_path(env.team_id(), src_path)
                if progress is None:
                    download_file(team_id, src_path, dst_path)
                else:
                    self.gui.download_progress.show()
                    with progress(
                        message="Downloading file from Team Files...",
                        total=file_info.sizeb,
                        unit="B",
                        unit_scale=True,
                    ) as pbar:
                        download_file(team_id, src_path, dst_path, pbar.update)
                logger.info(f"📥 File {basename} has been successfully downloaded from Team Files")
                logger.info(f"File {basename} path: {dst_path}")
        else:  # external url
            if not fs.dir_exists(os.path.dirname(dst_path)):
                fs.mkdir(os.path.dirname(dst_path))

            def download_external_file(url, save_path, progress=None):
                def download_content(save_path, progress_cb=None):
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            if progress is not None:
                                progress_cb(len(chunk))

                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(CaseInsensitiveDict(r.headers).get("Content-Length", "0"))
                    if progress is None:
                        download_content(save_path)
                    else:
                        with progress(
                            message="Downloading file from external URL",
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                        ) as pbar:
                            download_content(save_path, pbar.update)

            if progress is None:
                download_external_file(src_path, dst_path)
            else:
                self.gui.download_progress.show()
                download_external_file(src_path, dst_path, progress=progress)
            logger.info(f"📥 File {basename} has been successfully downloaded from external URL.")
            logger.info(f"File {basename} path: {dst_path}")
        return dst_path

    def _preprocess_models_list(self, models_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # fill skipped columns
        all_columns = []
        for model_dict in models_list:
            cols = model_dict.keys()
            all_columns.extend([col for col in cols if col not in all_columns])

        empty_cells = {}
        for col in all_columns:
            empty_cells[col] = []
        # fill empty cells by "-", write empty cells and set cells in column order
        for i in range(len(models_list)):
            model_dict = OrderedDict()
            for col in all_columns:
                if col not in models_list[i].keys():
                    model_dict[col] = "-"
                    empty_cells[col].append(True)
                else:
                    model_dict[col] = models_list[i][col]
                    empty_cells[col].append(False)
            models_list[i] = model_dict
        # remove empty columns
        for col, cells in empty_cells.items():
            if all(cells):
                for i, model_dict in enumerate(models_list):
                    del model_dict[col]

        return models_list

    # pylint: disable=method-hidden
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def load_model(self, **kwargs):
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def load_model_meta(self, model_tab: str, local_weights_path: str):
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def shutdown_model(self):
        self._model_served = False
        clean_up_cuda()
        logger.info("Model has been stopped")

    def _on_model_deployed(self):
        pass

    def get_classes(self) -> List[str]:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def get_info(self) -> Dict[str, Any]:
        num_classes = None
        classes = None
        try:
            classes = self.get_classes()
            num_classes = len(classes)
        except NotImplementedError:
            logger.warn(f"get_classes() function not implemented for {type(self)} object.")
        except AttributeError:
            logger.warn("Probably, get_classes() function not working without model deploy.")
        except Exception as exc:
            logger.warn("Unknown exception. Please, contact support")
            logger.exception(exc)

        if num_classes is None:
            logger.warn(f"get_classes() function return {classes}; skip classes processing.")

        return {
            "app_name": get_name_from_env(default="Neural Network Serving"),
            "session_id": self.task_id,
            "number_of_classes": num_classes,
            "sliding_window_support": self.sliding_window_mode,
            "videos_support": True,
            "async_video_inference_support": True,
            "tracking_on_videos_support": True,
            "async_image_inference_support": True,
            "tracking_algorithms": ["bot", "deepsort"],
        }

    # pylint: enable=method-hidden

    def get_human_readable_info(self, replace_none_with: Optional[str] = None):
        hr_info = {}
        info = self.get_info()

        for name, data in info.items():
            hr_name = name.replace("_", " ").capitalize()
            if data is None:
                hr_info[hr_name] = replace_none_with
            else:
                hr_info[hr_name] = data

        return hr_info

    @property
    def sliding_window_mode(self) -> Literal["basic", "advanced", "none"]:
        return self._sliding_window_mode

    @property
    def api(self) -> Api:
        if self._api is None:
            self._api = Api()
        return self._api

    @property
    def gui(self) -> GUI.InferenceGUI:
        return self._gui

    def _get_obj_class_shape(self):
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def model_meta(self) -> ProjectMeta:
        if self._model_meta is None:
            self.update_model_meta()
        return self._model_meta

    def update_model_meta(self):
        """
        Update model meta.
        Make sure `self._get_obj_class_shape()` method returns the correct shape.
        """
        colors = get_predefined_colors(len(self.get_classes()))
        classes = []
        for name, rgb in zip(self.get_classes(), colors):
            classes.append(ObjClass(name, self._get_obj_class_shape(), rgb))
        self._model_meta = ProjectMeta(classes)
        self._get_confidence_tag_meta()

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
        self,
        image_path: str,
        predictions: List[Prediction],
        classes_whitelist: Optional[List[str]] = None,
    ) -> Annotation:
        labels = []
        for prediction in predictions:
            if (
                not classes_whitelist in (None, "all")
                and prediction.class_name not in classes_whitelist
            ):
                continue
            label = self._create_label(prediction)
            if label is None:
                # for example empty mask
                continue
            if isinstance(label, list):
                labels.extend(label)
                continue
            labels.append(label)

        # create annotation with correct image resolution
        if isinstance(image_path, str):
            ann = Annotation.from_img_path(image_path)
        else:
            ann = Annotation(image_path.shape[:2])
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

    def _handle_error_in_async(self, uuid, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            inf_request = self._inference_requests.get(uuid, None)
            if inf_request is not None:
                inf_request["exception"] = str(e)
            logger.error(f"Error in {func.__name__} function: {e}", exc_info=True)
            raise e

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
            "Inferring image_path:",
            extra={"inference_mode": inference_mode, "path": image_path},
        )

        if inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced":
            predictions = self.predict_raw(image_path=image_path, settings=settings)
        else:
            predictions = self.predict(image_path=image_path, settings=settings)
        ann = self._predictions_to_annotation(
            image_path, predictions, classes_whitelist=settings.get("classes", None)
        )

        logger.debug(
            f"Inferring image_path done. pred_annotation:",
            extra=dict(w=ann.img_size[1], h=ann.img_size[0], n_labels=len(ann.labels)),
        )
        return ann

    @process_images_batch_sliding_window
    @process_images_batch_roi
    def _inference_images_batch(
        self,
        source: List,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ) -> List[Annotation]:
        inference_mode = settings.get("inference_mode", "full_image")
        logger.debug(
            "Inferring images batch:",
            extra={"inference_mode": inference_mode, "items": len(source)},
        )
        if len(source) == 0:
            return []

        def _predict(source: List, predict_func: Callable):
            temp_dir = None
            if isinstance(source[0], np.ndarray):
                temp_dir = os.path.join(get_data_dir(), rand_str(10))
                fs.mkdir(temp_dir)
                images_paths = [os.path.join(temp_dir, f"{rand_str(10)}.png") for _ in source]
                for img_path, img in zip(images_paths, source):
                    sly_image.write(img_path, img)
            else:
                images_paths = source
            predictions = [
                predict_func(image_path=img_path, settings=settings) for img_path in images_paths
            ]
            if temp_dir is not None:
                fs.remove_dir(temp_dir)
            return predictions

        if inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced":
            if type(self).predict_batch_raw == Inference.predict_batch_raw:
                predictions = _predict(source, self.predict_raw)
            else:
                predictions = self.predict_batch_raw(source=source, settings=settings)
        else:
            if type(self).predict_batch == Inference.predict_batch:
                predictions = _predict(source, self.predict)
            else:
                predictions = self.predict_batch(source=source, settings=settings)

        anns = [
            self._predictions_to_annotation(
                image_path, prediction, classes_whitelist=settings.get("classes", None)
            )
            for image_path, prediction in zip(source, predictions)
        ]

        logger.debug(
            f"Inferring images batch done. pred_annotations:",
            extra={
                "annotations": [
                    dict(w=ann.img_size[1], h=ann.img_size[0], n_labels=len(ann.labels))
                    for ann in anns
                ]
            },
        )
        return anns

    # pylint: disable=method-hidden
    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def predict_batch(self, source: List, settings: Dict[str, Any]) -> List[List[Prediction]]:
        """Predict batch of images. source can be either list of image paths or list of numpy arrays.
        If source is a list of numpy arrays, it should be in BGR format"""
        raise NotImplementedError("Have to be implemented in child class")

    def predict_batch_raw(self, source: List, settings: Dict[str, Any]) -> List[List[Prediction]]:
        """Predict batch of images. source can be either list of image paths or list of numpy arrays.
        If source is a list of numpy arrays, it should be in BGR format"""
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    # pylint: enable=method-hidden
    def _get_inference_settings(self, state: dict):
        settings = state.get("settings", {})
        if settings is None:
            settings = {}
        if "rectangle" in state.keys():
            settings["rectangle"] = state["rectangle"]
        settings["sliding_window_mode"] = self.sliding_window_mode

        for key, value in self.custom_inference_settings_dict.items():
            if key not in settings:
                logger.debug(
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
        classes_whitelist: Optional[List[str]] = None,
    ):
        image = sly_image.read(image_path)
        ann = self._predictions_to_annotation(image_path, predictions, classes_whitelist)
        ann.draw_pretty(
            bitmap=image,
            thickness=thickness,
            output_path=vis_path,
            fill_rectangles=False,
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
        if type(self).predict_batch == Inference.predict_batch:
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
        else:
            images = [sly_image.read_bytes(file.file.read()) for file in files]
            return self._inference_images_dir(images, state)

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
        results = []
        data_to_return = {}
        if type(self).predict_batch == Inference.predict_batch:
            n_imgs = len(img_paths)
            for i, image_path in enumerate(img_paths):
                data_to_return = {}
                logger.debug(f"Inferring image {i+1}/{n_imgs}.", extra={"path": image_path})
                ann = self._inference_image_path(
                    image_path=image_path,
                    settings=settings,
                    data_to_return=data_to_return,
                )
                results.append({"annotation": ann.to_json(), "data": data_to_return})
        else:
            anns = self._inference_images_batch(
                source=img_paths,
                settings=settings,
                data_to_return=data_to_return,
            )
            for i, ann in enumerate(anns):
                data = {}
                if "slides" in data_to_return:
                    data = data_to_return["slides"][i]
                results.append(
                    {
                        "annotation": ann.to_json(),
                        "data": data,
                    }
                )
        return results

    def _inference_image_id(self, api: Api, state: dict, async_inference_request_uuid: str = None):
        logger.debug("Inferring image_id...", extra={"state": state})
        settings = self._get_inference_settings(state)
        upload = state.get("upload", False)
        image_id = state["image_id"]
        image_info = api.image.get_info_by_id(image_id)
        image_path = os.path.join(get_data_dir(), f"{rand_str(10)}_{image_info.name}")
        api.image.download_path(image_id, image_path)
        logger.debug("Inference settings:", extra=settings)
        logger.debug(
            "Image info:",
            extra={"id": image_id, "w": image_info.width, "h": image_info.height},
        )
        logger.debug(f"Downloaded path: {image_path}")

        inference_request = {}
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

        data_to_return = {}
        ann = self._inference_image_path(
            image_path=image_path,
            settings=settings,
            data_to_return=data_to_return,
        )
        fs.silent_remove(image_path)

        if upload:
            ds_info = api.dataset.get_info_by_id(image_info.dataset_id, raise_error=True)
            output_project_id = ds_info.project_id
            output_project_meta = self.cache.get_project_meta(api, output_project_id)
            logger.debug("Merging project meta...")

            output_project_meta, ann, meta_changed = update_meta_and_ann(output_project_meta, ann)
            if meta_changed:
                output_project_meta = api.project.update_meta(
                    output_project_id, output_project_meta
                )
                self.cache.set_project_meta(output_project_id, output_project_meta)

            logger.debug(
                "Uploading annotation...",
                extra={
                    "image_id": image_id,
                    "dataset_id": ds_info.id,
                    "project_id": output_project_id,
                },
            )
            api.annotation.upload_ann(image_id, ann)

        result = {"annotation": ann.to_json(), "data": data_to_return}
        if async_inference_request_uuid is not None and ann is not None:
            inference_request["result"] = result
        return result

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
        n_frames = state.get("framesCount", video_info.frames_count)
        start_frame_index = state.get("startFrameIndex", 0)
        direction = state.get("direction", "forward")
        logger.debug(
            f"Video info:",
            extra=dict(
                w=video_info.frame_width,
                h=video_info.frame_height,
                start_frame_index=start_frame_index,
                n_frames=n_frames,
            ),
        )
        tracking = state.get("tracker", None)

        preparing_progress = {"current": 0, "total": 1}
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
            inference_request["preparing_progress"]["total"] = n_frames
            preparing_progress = inference_request["preparing_progress"]

        # progress
        preparing_progress["status"] = "download_video"
        preparing_progress["current"] = 0
        preparing_progress["total"] = int(video_info.file_meta["size"])

        def _progress_cb(chunk_size):
            preparing_progress["current"] += chunk_size

        self.cache.download_video(api, video_info.id, return_images=False, progress_cb=_progress_cb)
        preparing_progress["status"] = "inference"

        settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=settings)

        logger.debug(f"Total frames to infer: {n_frames}")

        if tracking == "bot":
            from supervisely.nn.tracker import BoTTracker

            tracker = BoTTracker(state)
        elif tracking == "deepsort":
            from supervisely.nn.tracker import DeepSortTracker

            tracker = DeepSortTracker(state)
        else:
            if tracking is not None:
                logger.warn(f"Unknown tracking type: {tracking}. Tracking is disabled.")
            tracker = None

        results = []
        batch_size = 16
        tracks_data = {}
        direction = 1 if direction == "forward" else -1
        for batch in batched(
            range(start_frame_index, start_frame_index + direction * n_frames, direction),
            batch_size,
        ):
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
            logger.debug(
                f"Inferring frames {batch[0]}-{batch[-1]}:",
            )
            frames = self.cache.download_frames(api, video_info.id, batch)
            data_to_return = {}
            anns = self._inference_images_batch(
                source=frames,
                settings=settings,
                data_to_return=data_to_return,
            )
            if tracker is not None:
                for frame_index, frame, ann in zip(batch, frames, anns):
                    tracks_data = tracker.update(frame, ann, frame_index, tracks_data)
            batch_results = []
            for i, ann in enumerate(anns):
                data = {}
                if "slides" in data_to_return:
                    data = data_to_return["slides"][i]
                batch_results.append(
                    {
                        "annotation": ann.to_json(),
                        "data": data,
                    }
                )
            results.extend(batch_results)
            if async_inference_request_uuid is not None:
                sly_progress.iters_done(len(batch))
                inference_request["pending_results"].extend(batch_results)
            logger.debug(f"Frames {batch[0]}-{batch[-1]} done.")
        video_ann_json = None
        if tracker is not None:
            frames = self.cache.download_frames(
                api, video_info.id, range(start_frame_index, start_frame_index + n_frames)
            )
            video_ann_json = tracker.get_annotation(
                tracks_data, (video_info.frame_height, video_info.frame_width), n_frames
            ).to_json()
        result = {"ann": results, "video_ann": video_ann_json}
        if async_inference_request_uuid is not None and len(results) > 0:
            inference_request["result"] = result.copy()
        return result

    def _inference_project_id(
        self,
        api: Api,
        state: dict,
        project_info: ProjectInfo = None,
        async_inference_request_uuid: str = None,
    ):
        """Inference project images.
        If "output_project_id" in state, upload images and annotations to the output project.
        If "output_project_id" equal to source project id, upload annotations to the source project.
        If "output_project_id" is None, write annotations to inference request object.
        """
        logger.debug("Inferring project...", extra={"state": state})
        if project_info is None:
            project_info = api.project.get_info_by_id(state["projectId"])
        dataset_ids = state.get("dataset_ids", None)
        cache_project_on_model = state.get("cache_project_on_model", False)
        batch_size = state.get("batch_size", 16)

        datasets_infos = api.dataset.get_list(project_info.id, recursive=True)
        if dataset_ids is not None:
            datasets_infos = [ds_info for ds_info in datasets_infos if ds_info.id in dataset_ids]

        # progress
        preparing_progress = {"current": 0, "total": 1}
        preparing_progress["status"] = "download_info"
        preparing_progress["current"] = 0
        preparing_progress["total"] = len(datasets_infos)
        progress_cb = None
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
            sly_progress.total = sum([ds_info.items_count for ds_info in datasets_infos])

            inference_request["preparing_progress"]["total"] = len(datasets_infos)
            preparing_progress = inference_request["preparing_progress"]

            if cache_project_on_model:
                progress_cb = sly_progress.iters_done
                preparing_progress["total"] = sly_progress.total
                preparing_progress["status"] = "download_project"

        output_project_id = state.get("output_project_id", None)
        output_project_meta = None
        if output_project_id is not None:
            logger.debug("Merging project meta...")
            output_project_meta = ProjectMeta.from_json(api.project.get_meta(output_project_id))
            changed = False
            for obj_class in self.model_meta.obj_classes:
                if output_project_meta.obj_classes.get(obj_class.name, None) is None:
                    output_project_meta = output_project_meta.add_obj_class(obj_class)
                    changed = True
            for tag_meta in self.model_meta.tag_metas:
                if output_project_meta.tag_metas.get(tag_meta.name, None) is None:
                    output_project_meta = output_project_meta.add_tag_meta(tag_meta)
                    changed = True
            if changed:
                output_project_meta = api.project.update_meta(
                    output_project_id, output_project_meta
                )

        if cache_project_on_model:
            download_to_cache(api, project_info.id, datasets_infos, progress_cb=progress_cb)

        images_infos_dict = {}
        for dataset_info in datasets_infos:
            images_infos_dict[dataset_info.id] = api.image.get_list(dataset_info.id)
            if not cache_project_on_model:
                preparing_progress["current"] += 1

        preparing_progress["status"] = "inference"
        preparing_progress["current"] = 0

        def _download_images(datasets_infos: List[DatasetInfo]):
            for dataset_info in datasets_infos:
                image_ids = [image_info.id for image_info in images_infos_dict[dataset_info.id]]
                with ThreadPoolExecutor(batch_size) as executor:
                    for image_id in image_ids:
                        executor.submit(
                            self.cache.download_image,
                            api,
                            image_id,
                        )

        if not cache_project_on_model:
            # start downloading in parallel
            threading.Thread(target=_download_images, args=[datasets_infos], daemon=True).start()

        def _upload_results_to_source(results: List[Dict]):
            nonlocal output_project_meta
            for result in results:
                image_id = result["image_id"]
                ann = Annotation.from_json(result["annotation"], self.model_meta)
                output_project_meta, ann, meta_changed = update_meta_and_ann(
                    output_project_meta, ann
                )
                if meta_changed:
                    output_project_meta = api.project.update_meta(
                        project_info.id, output_project_meta
                    )
                ann = update_classes(api, ann, output_project_meta, output_project_id)
                api.annotation.append_labels(image_id, ann.labels)
                if async_inference_request_uuid is not None:
                    sly_progress.iters_done(1)
                    inference_request["pending_results"].append(
                        {
                            "annotation": None,  # to less response size
                            "data": None,  # to less response size
                            "image_id": image_id,
                            "image_name": result["image_name"],
                            "dataset_id": result["dataset_id"],
                        }
                    )

        new_dataset_id = {}

        def _get_or_create_new_dataset(output_project_id, src_dataset_id):
            """Copy dataset in output project if not exists and return its id"""
            if src_dataset_id in new_dataset_id:
                return new_dataset_id[src_dataset_id]
            dataset_info = api.dataset.get_info_by_id(src_dataset_id)
            output_dataset_id = api.dataset.copy(
                output_project_id, src_dataset_id, dataset_info.name, change_name_if_conflict=True
            ).id
            new_dataset_id[src_dataset_id] = output_dataset_id
            return output_dataset_id

        def _upload_results_to_other(results: List[Dict]):
            nonlocal output_project_meta
            if len(results) == 0:
                return
            src_dataset_id = results[0]["dataset_id"]
            dataset_id = _get_or_create_new_dataset(output_project_id, src_dataset_id)
            image_names = [result["image_name"] for result in results]
            image_infos = api.image.get_list(
                dataset_id, filters=[{"field": "name", "operator": "in", "value": image_names}]
            )
            meta_changed = False
            anns = []
            for result in results:
                ann = Annotation.from_json(result["annotation"], self.model_meta)
                output_project_meta, ann, c = update_meta_and_ann(output_project_meta, ann)
                meta_changed = meta_changed or c
                anns.append(ann)
            if meta_changed:
                api.project.update_meta(output_project_id, output_project_meta)

            # upload in batches to update progress with each batch
            # api.annotation.upload_anns() uploads in same batches anyways
            for batch in batched(list(zip(anns, results, image_infos))):
                batch_anns, batch_results, batch_image_infos = zip(*batch)
                api.annotation.upload_anns(
                    img_ids=[info.id for info in batch_image_infos],
                    anns=batch_anns,
                )
                if async_inference_request_uuid is not None:
                    sly_progress.iters_done(len(batch_results))
                    inference_request["pending_results"].extend(
                        [{**result, "annotation": None, "data": None} for result in batch_results]
                    )

        def _add_results_to_request(results: List[Dict]):
            if async_inference_request_uuid is None:
                return
            inference_request["pending_results"].extend(results)
            sly_progress.iters_done(len(results))

        def _upload_loop(q: Queue, stop_event: threading.Event, api: Api, upload_f: Callable):
            try:
                while True:
                    items = []
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        ds_batches = {}
                        for batch in items:
                            if len(batch) == 0:
                                continue
                            ds_batches.setdefault(batch[0].get("dataset_id"), []).extend(batch)
                        for _, joined_batch in ds_batches.items():
                            upload_f(joined_batch)
                        continue
                    if stop_event.is_set():
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                raise

        if output_project_id is None:
            upload_f = _add_results_to_request
        elif output_project_id != project_info.id:
            upload_f = _upload_results_to_other
        else:
            upload_f = _upload_results_to_source

        upload_queue = Queue()
        stop_upload_event = threading.Event()
        upload_thread = threading.Thread(
            target=_upload_loop,
            args=[upload_queue, stop_upload_event, api, upload_f],
            daemon=True,
        )
        upload_thread.start()

        settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=settings)
        results = []
        data_to_return = {}
        stop = False
        try:
            for dataset_info in datasets_infos:
                if stop:
                    break
                for images_infos_batch in batched(
                    images_infos_dict[dataset_info.id], batch_size=batch_size
                ):
                    if (
                        async_inference_request_uuid is not None
                        and inference_request["cancel_inference"] is True
                    ):
                        logger.debug(
                            f"Cancelling inference project...",
                            extra={"inference_request_uuid": async_inference_request_uuid},
                        )
                        results = []
                        stop = True
                        break
                    if cache_project_on_model:
                        images_nps, _ = zip(
                            *read_from_cached_project(
                                project_info.id,
                                dataset_info.name,
                                [ii.name for ii in images_infos_batch],
                            )
                        )
                    else:
                        images_nps = self.cache.download_images(
                            api,
                            dataset_info.id,
                            [info.id for info in images_infos_batch],
                            return_images=True,
                        )
                    anns = self._inference_images_batch(
                        source=images_nps,
                        settings=settings,
                        data_to_return=data_to_return,
                    )
                    batch_results = []
                    for i, ann in enumerate(anns):
                        data = {}
                        if "slides" in data_to_return:
                            data = data_to_return["slides"][i]
                        batch_results.append(
                            {
                                "annotation": ann.to_json(),
                                "data": data,
                                "image_id": images_infos_batch[i].id,
                                "image_name": images_infos_batch[i].name,
                                "dataset_id": dataset_info.id,
                            }
                        )
                    results.extend(batch_results)
                    upload_queue.put(batch_results)
        except Exception:
            stop_upload_event.set()
            upload_thread.join()
            raise
        if async_inference_request_uuid is not None and len(results) > 0:
            inference_request["result"] = {"ann": results}
        stop_upload_event.set()
        upload_thread.join()
        return results

    def _on_inference_start(self, inference_request_uuid):
        inference_request = {
            "progress": Progress("Inferring model...", total_cnt=1),
            "is_inferring": True,
            "cancel_inference": False,
            "result": None,
            "pending_results": [],
            "preparing_progress": {"current": 0, "total": 1},
            "exception": None,
        }
        self._inference_requests[inference_request_uuid] = inference_request

    def _on_inference_end(self, future, inference_request_uuid):
        logger.debug("callback: on_inference_end()")
        inference_request = self._inference_requests.get(inference_request_uuid)
        if inference_request is not None:
            inference_request["is_inferring"] = False

    def _check_serve_before_call(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._model_served is True:
                return func(*args, **kwargs)
            else:
                msg = (
                    "The model has not yet been deployed. "
                    "Please select the appropriate model in the UI and press the 'Serve' button. "
                    "If this app has no GUI, it signifies that 'load_on_device' was never called."
                )
                # raise DialogWindowError(title="Call undeployed model.", description=msg)
                raise RuntimeError(msg)

        return wrapper

    def _set_served_callback(self):
        self._model_served = True

    def serve(self):
        if not self._use_gui:
            Progress("Deploying model ...", 1)

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
            self._task_id = env.task_id() if is_production() else None

        if isinstance(self.gui, GUI.InferenceGUI):
            self._app = Application(layout=self.get_ui())
        elif isinstance(self.gui, GUI.ServingGUI):
            serving_layout = self.get_ui()

            self._user_layout_card = Card(
                title="Select Model",
                description="Select the model to deploy and press the 'Serve' button.",
                content=self._user_layout,
                lock_message="Model is deployed. To change the model, stop the serving first.",
            )
            self._api_request_model_info = Editor(
                height_lines=12,
                language_mode="json",
                readonly=True,
                restore_default_button=False,
                auto_format=True,
            )
            self._api_request_model_layout = Card(
                title="Model was deployed from API request with the following settings",
                content=self._api_request_model_info,
            )
            self._api_request_model_layout.hide()
            layout = Container(
                [self._user_layout_card, self._api_request_model_layout, serving_layout]
            )
            self._app = Application(layout=layout)
        else:
            self._app = Application(layout=self.get_ui())

        server = self._app.get_server()

        @call_on_autostart()
        def autostart_func():
            gpu_count = get_gpu_count()
            if gpu_count > 1:
                # run autostart after 5 min
                def delayed_autostart():
                    logger.debug("Found more than one GPU, autostart will be delayed.")
                    time.sleep(self._autostart_delay_time)
                    if not self._model_served:
                        logger.debug("Deploying the model via autostart...")
                        self.gui.deploy_with_current_params()

                self._executor.submit(delayed_autostart)
            else:
                # run autostart immediately
                self.gui.deploy_with_current_params()

        if not self._use_gui:
            Progress("Model deployed", 1).iter_done_report()
        else:
            autostart_func()

        @server.post(f"/get_session_info")
        @self._check_serve_before_call
        def get_session_info(response: Response):
            return self.get_info()

        @server.post("/get_custom_inference_settings")
        def get_custom_inference_settings():
            return {"settings": self.custom_inference_settings}

        @server.post("/get_output_classes_and_tags")
        def get_output_classes_and_tags():
            return self.model_meta.to_json()

        @server.post("/inference_image_id")
        def inference_image_id(request: Request):
            logger.debug(f"'inference_image_id' request in json format:{request.state.state}")
            return self._inference_image_id(request.state.api, request.state.state)

        @server.post("/inference_image_url")
        def inference_image_url(request: Request):
            logger.debug(f"'inference_image_url' request in json format:{request.state.state}")
            return self._inference_image_url(request.state.api, request.state.state)

        @server.post("/inference_batch_ids")
        def inference_batch_ids(request: Request):
            logger.debug(f"'inference_batch_ids' request in json format:{request.state.state}")
            return self._inference_batch_ids(request.state.api, request.state.state)

        @server.post("/inference_video_id")
        def inference_video_id(request: Request):
            logger.debug(f"'inference_video_id' request in json format:{request.state.state}")
            return self._inference_video_id(request.state.api, request.state.state)

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

        @server.post("/inference_image_id_async")
        def inference_image_id_async(request: Request):
            logger.debug(f"'inference_image_id_async' request in json format:{request.state.state}")
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._inference_image_id,
                request.state.api,
                request.state.state,
                inference_request_uuid,
            )
            end_callback = partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            logger.debug(
                "Inference has scheduled from 'inference_image_id_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post("/inference_video_id_async")
        def inference_video_id_async(request: Request):
            logger.debug(f"'inference_video_id_async' request in json format:{request.state.state}")
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._inference_video_id,
                request.state.api,
                request.state.state,
                inference_request_uuid,
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

        @server.post("/inference_project_id_async")
        def inference_project_id_async(request: Request):
            logger.debug(
                f"'inference_project_id_async' request in json format:{request.state.state}"
            )
            project_id = request.state.state["projectId"]
            project_info = request.state.api.project.get_info_by_id(project_id)
            if project_info.type != str(ProjectType.IMAGES):
                raise ValueError("Only images projects are supported.")

            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._inference_project_id,
                request.state.api,
                request.state.state,
                project_info,
                inference_request_uuid,
            )
            end_callback = partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            logger.debug(
                "Inference has scheduled from 'inference_project_id_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post(f"/get_inference_progress")
        def get_inference_progress(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid].copy()
            inference_request["progress"] = _convert_sly_progress_to_dict(
                inference_request["progress"]
            )

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request
            )
            logger.debug(
                f"Sending inference progress with uuid:",
                extra=log_extra,
            )

            # Ger rid of `pending_results` to less response size
            inference_request["pending_results"] = []
            return inference_request

        @server.post(f"/pop_inference_results")
        def pop_inference_results(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            # Copy results
            inference_request = self._inference_requests[inference_request_uuid].copy()
            inference_request["pending_results"] = inference_request["pending_results"].copy()

            # Clear the queue `pending_results`
            self._inference_requests[inference_request_uuid]["pending_results"].clear()

            inference_request["progress"] = _convert_sly_progress_to_dict(
                inference_request["progress"]
            )

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request
            )
            logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
            return inference_request

        @server.post(f"/get_inference_result")
        def get_inference_result(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid].copy()

            inference_request["progress"] = _convert_sly_progress_to_dict(
                inference_request["progress"]
            )

            # Logging
            log_extra = _get_log_extra_for_inference_request(
                inference_request_uuid, inference_request
            )
            logger.debug(
                f"Sending inference result with uuid:",
                extra=log_extra,
            )

            return inference_request["result"]

        @server.post(f"/stop_inference")
        def stop_inference(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": "Error: 'inference_request_uuid' is required.",
                    "success": False,
                }
            inference_request = self._inference_requests[inference_request_uuid]
            inference_request["cancel_inference"] = True
            return {"message": "Inference will be stopped.", "success": True}

        @server.post(f"/clear_inference_request")
        def clear_inference_request(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": "Error: 'inference_request_uuid' is required.",
                    "success": False,
                }
            del self._inference_requests[inference_request_uuid]
            logger.debug("Removed an inference request:", extra={"uuid": inference_request_uuid})
            return {"success": True}

        @server.post(f"/get_preparing_progress")
        def get_preparing_progress(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self._inference_requests[inference_request_uuid].copy()
            return inference_request["preparing_progress"]

        @server.post("/get_deploy_settings")
        def _get_deploy_settings(response: Response, request: Request):
            """
            Get deploy settings for the model. Works only for the Sphinx docstring format.
            """
            load_model_method = getattr(self, "load_model")
            method_signature = inspect.signature(load_model_method)
            docstring = inspect.getdoc(load_model_method) or ""
            doc_lines = docstring.split("\n")

            param_docs = {}
            param_type = {}
            for line in doc_lines:
                if line.startswith(":param"):
                    name = line.split(":")[1].strip().split()[1]
                    doc = line.replace(f":param {name}:", "").strip()
                    param_docs[name] = doc
                if line.startswith(":type"):
                    name = line.split(":")[1].strip().split()[1]
                    type = line.replace(f":type {name}:", "").strip()
                    param_type[name] = type

            args_details = []
            for name, parameter in method_signature.parameters.items():
                if name == "self":
                    continue
                arg_type = param_type.get(name, None)
                default = (
                    parameter.default if parameter.default != inspect.Parameter.empty else None
                )
                doc = param_docs.get(name, None)
                args_details.append(
                    {"name": name, "type": arg_type, "default": default, "doc": doc}
                )

            return args_details

        @server.post("/deploy_from_api")
        def _deploy_from_api(response: Response, request: Request):
            try:
                if self._model_served:
                    self.shutdown_model()
                state = request.state.state
                deploy_params = state["deploy_params"]
                self.load_model(**deploy_params)
                self.update_gui(self._model_served)
                self.set_params_to_gui(deploy_params)

                # update to set correct device
                device = deploy_params.get("device", "cpu")
                self.gui.set_deployed(device)

                self.gui.show_deployed_model_info(self)
                self._model_served = True
                return {"result": "model was successfully deployed"}
            except Exception as e:
                self.gui._success_label.hide()
                raise e

        @server.post("/is_deployed")
        def _is_deployed(response: Response, request: Request):
            return {
                "deployed": self._model_served,
                "description:": "Model is ready to receive requests",
            }


def _get_log_extra_for_inference_request(inference_request_uuid, inference_request: dict):
    log_extra = {
        "uuid": inference_request_uuid,
        "progress": inference_request["progress"],
        "is_inferring": inference_request["is_inferring"],
        "cancel_inference": inference_request["cancel_inference"],
        "has_result": inference_request["result"] is not None,
        "pending_results": len(inference_request["pending_results"]),
    }
    return log_extra


def _convert_sly_progress_to_dict(sly_progress: Progress):
    return {
        "current": sly_progress.current,
        "total": sly_progress.total,
    }


def _create_notify_after_complete_decorator(
    msg: str,
    *,
    arg_pos: Optional[int] = None,
    arg_key: Optional[str] = None,
):
    """
    Decorator to log message after wrapped function complete.

    :param msg: info message
    :type msg: str
    :param arg_pos: position of argument in `args` to insert in message
    :type arg_pos: Optional[int]
    :param arg_key: key of argument in `kwargs` to insert in message.
        If an argument can be both positional and keyword,
        it is preferable to declare both 'arg_pos' and 'arg_key'
    :type arg_key: Optional[str]
    :Usage example:

     .. code-block:: python

        @_create_notify_after_complete_decorator("Print arg1: %s", arg_pos=0)
        def wrapped_function(arg1, kwarg1)
            return

        wrapped_function("pos_arg", kwarg1="key_arg")
        # Info    2023.07.04 11:37:59     Print arg1: pos_arg
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if arg_key is not None and arg_key in kwargs:
                arg = kwargs[arg_key]
                logger.info(msg, str(arg))
            elif arg_pos is not None and arg_pos < len(args):
                arg = args[arg_pos]
                logger.info(msg, str(arg))
            else:
                logger.info(msg, "some")
            return result

        return wrapper

    return decorator


LOAD_ON_DEVICE_DECORATOR = _create_notify_after_complete_decorator(
    "✅ Model has been successfully deployed on %s device",
    arg_pos=1,
    arg_key="device",
)

LOAD_MODEL_DECORATOR = _create_notify_after_complete_decorator(
    "✅ Model has been successfully deployed on %s device",
    arg_pos=1,
    arg_key="device",
)


def get_gpu_count():
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        gpu_count = len(re.findall(r"GPU \d+:", nvidia_smi_output))
        return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warn("Calling nvidia-smi caused a error: {exc}. Assume there is no any GPU.")
        return 0


def clean_up_cuda():
    try:
        # torch may not be installed
        import gc

        # pylint: disable=import-error
        # pylint: disable=method-hidden
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("Error in clean_up_cuda.", exc_info=True)


def update_meta_and_ann(meta: ProjectMeta, ann: Annotation):
    """Update project meta and annotation to match each other
    If obj class or tag meta from annotation conflicts with project meta
    add suffix to obj class or tag meta.
    Return tuple of updated project meta, annotation and boolean flag if meta was changed."""
    obj_classes_suffixes = {"_nn"}
    tag_meta_suffixes = {"_nn"}
    ann_obj_classes = {}
    ann_tag_metas = {}
    meta_changed = False

    # get all obj classes and tag metas from annotation
    for label in ann.labels:
        ann_obj_classes[label.obj_class.name] = label.obj_class
        for tag in label.tags:
            ann_tag_metas[tag.meta.name] = tag.meta
    for tag in ann.img_tags:
        ann_tag_metas[tag.meta.name] = tag.meta

    # check if obj classes are in project meta
    # if not, add them with suffix
    changed_obj_classes = {}
    for ann_obj_class in ann_obj_classes.values():
        if meta.get_obj_class(ann_obj_class.name) is None:
            meta = meta.add_obj_class(ann_obj_class)
            meta_changed = True
        elif meta.get_obj_class(ann_obj_class.name).geometry_type != ann_obj_class.geometry_type:
            found = False
            for suffix in obj_classes_suffixes:
                new_obj_class_name = ann_obj_class.name + suffix
                meta_obj_class = meta.get_obj_class(new_obj_class_name)
                if meta_obj_class is None:
                    new_obj_class = ann_obj_class.clone(name=new_obj_class_name)
                    meta = meta.add_obj_class(new_obj_class)
                    meta_changed = True
                    changed_obj_classes[ann_obj_class.name] = new_obj_class
                    found = True
                    break
                if meta_obj_class.geometry_type == ann_obj_class.geometry_type:
                    changed_obj_classes[ann_obj_class.name] = meta_obj_class
                    found = True
                    break
            if not found:
                raise ValueError(f"Can't add obj class {ann_obj_class.name} to project meta")

    # check if tag metas are in project meta
    # if not, add them with suffix
    changed_tag_metas = {}
    for tag_meta in ann_tag_metas.values():
        if meta.get_tag_meta(tag_meta.name) is None:
            meta = meta.add_tag_meta(tag_meta)
            meta_changed = True
        elif not meta.get_tag_meta(tag_meta.name).is_compatible(tag_meta):
            found = False
            for suffix in tag_meta_suffixes:
                new_tag_meta_name = tag_meta.name + suffix
                meta_tag_meta = meta.get_tag_meta(new_tag_meta_name)
                if meta_tag_meta is None:
                    new_tag_meta = tag_meta.clone(name=new_tag_meta_name)
                    meta = meta.add_tag_meta(new_tag_meta)
                    changed_tag_metas[tag_meta.name] = new_tag_meta
                    meta_changed = True
                    found = True
                    break
                if meta_tag_meta.is_compatible(tag_meta):
                    changed_tag_metas[tag_meta.name] = meta_tag_meta
                    found = True
                    break
            if not found:
                raise ValueError(f"Can't add tag meta {tag_meta.name} to project meta")

    labels = []
    for label in ann.labels:
        if label.obj_class.name in changed_obj_classes:
            label = label.clone(obj_class=changed_obj_classes[label.obj_class.name])

        label_tags = []
        for tag in label.tags:
            if tag.meta.name in changed_tag_metas:
                label_tags.append(tag.clone(meta=changed_tag_metas[tag.meta.name]))
            else:
                label_tags.append(tag)

        labels.append(label.clone(tags=TagCollection(label_tags)))
    img_tags = []
    for tag in ann.img_tags:
        if tag.meta.name in changed_tag_metas:
            img_tags.append(tag.clone(meta=changed_tag_metas[tag.meta.name]))
        else:
            img_tags.append(tag)

    ann = ann.clone(labels=labels, img_tags=TagCollection(img_tags))
    return meta, ann, meta_changed


def update_classes(api: Api, ann: Annotation, meta: ProjectMeta, project_id: int):
    labels = []
    for label in ann.labels:
        if label.obj_class.sly_id is None:
            obj_class = meta.get_obj_class(label.obj_class.name)
            if obj_class.sly_id is None:
                meta = api.project.update_meta(project_id, meta)
                obj_class = meta.get_obj_class(label.obj_class.name)
            labels.append(label.clone(obj_class=obj_class))
        else:
            labels.append(label)
    return ann.clone(labels=labels)
