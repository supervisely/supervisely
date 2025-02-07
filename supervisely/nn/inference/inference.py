import argparse
import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial, wraps
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
import requests
import uvicorn
import yaml
from fastapi import Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from requests.structures import CaseInsensitiveDict

import supervisely.app.development as sly_app_development
import supervisely.imaging.image as sly_image
import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
import supervisely.nn.inference.gui as GUI
from supervisely import DatasetInfo, ProjectInfo, VideoAnnotation, batched
from supervisely._utils import (
    add_callback,
    get_filename_from_headers,
    get_or_create_event_loop,
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
from supervisely.api.app_api import WorkflowMeta, WorkflowSettings
from supervisely.api.image_api import ImageInfo
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
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.imaging.color import get_predefined_colors
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.prediction_dto import Prediction
from supervisely.nn.utils import (
    CheckpointInfo,
    DeployInfo,
    ModelPrecision,
    ModelSource,
    RuntimeType,
)
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
    FRAMEWORK_NAME: str = None
    """Name of framework to register models in Supervisely"""
    MODELS: str = None
    """Path to file with list of models"""
    APP_OPTIONS: str = None
    """Path to file with app options"""
    DEFAULT_BATCH_SIZE: str = 16
    """Default batch size for inference"""
    INFERENCE_SETTINGS: str = None
    """Path to file with custom inference settings"""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[
            Union[Dict[str, Any], str]
        ] = None,  # dict with settings or path to .yml file
        sliding_window_mode: Optional[Literal["basic", "advanced", "none"]] = "basic",
        use_gui: Optional[bool] = False,
        multithread_inference: Optional[bool] = True,
        use_serving_gui_template: Optional[bool] = False,
    ):

        self._args, self._is_local_deploy = self._parse_local_deploy_args()

        if model_dir is None:
            if self._is_local_deploy is True:
                try:
                    model_dir = get_data_dir()
                except:
                    model_dir = Path("~/.cache/supervisely/app_data").expanduser()
            else:
                model_dir = os.path.join(get_data_dir(), "models")
        sly_fs.mkdir(model_dir)

        self.device: str = None
        self.runtime: str = None
        self.model_precision: str = None
        self.model_source: str = None
        self.checkpoint_info: CheckpointInfo = None
        self.max_batch_size: int = None  # set it only if a model has a limit on the batch size
        self.classes: List[str] = None
        self._model_dir = model_dir
        self._model_served = False
        self._deploy_params: dict = None
        self._model_meta = None
        self._confidence = "confidence"
        self._app: Application = None
        self._api: Api = None
        self._task_id = None
        self._sliding_window_mode = sliding_window_mode
        self._autostart_delay_time = 5 * 60  # 5 min
        self._tracker = None
        self._hardware: str = None
        self.pretrained_models = self._load_models_json_file(self.MODELS) if self.MODELS else None
        if custom_inference_settings is None:
            if self.INFERENCE_SETTINGS is not None:
                custom_inference_settings = self.INFERENCE_SETTINGS
            else:
                logger.debug("Custom inference settings are not provided.")
                custom_inference_settings = {}
        if isinstance(custom_inference_settings, str):
            if sly_fs.file_exists(custom_inference_settings):
                with open(custom_inference_settings, "r") as f:
                    custom_inference_settings = f.read()
            else:
                raise FileNotFoundError(f"{custom_inference_settings} file not found.")
        self._custom_inference_settings = custom_inference_settings

        self._use_gui = use_gui
        self._use_serving_gui_template = use_serving_gui_template
        self._gui = None
        self._uvicorn_server = None

        self.load_on_device = LOAD_ON_DEVICE_DECORATOR(self.load_on_device)
        self.load_on_device = add_callback(self.load_on_device, self._set_served_callback)

        self.load_model = LOAD_MODEL_DECORATOR(self.load_model)

        if self._is_local_deploy:
            # self._args = self._parse_local_deploy_args()
            self._use_gui = False
            deploy_params, need_download = self._get_deploy_params_from_args()
            if need_download:
                local_model_files = self._download_model_files(
                    deploy_params["model_source"], deploy_params["model_files"], False
                )
                deploy_params["model_files"] = local_model_files
            self._load_model_headless(**deploy_params)

        if self._use_gui:
            initialize_custom_gui_method = getattr(self, "initialize_custom_gui", None)
            original_initialize_custom_gui_method = getattr(
                Inference, "initialize_custom_gui", None
            )
            if self._use_serving_gui_template:
                if self.FRAMEWORK_NAME is None:
                    raise ValueError("FRAMEWORK_NAME is not defined")
                self._gui = GUI.ServingGUITemplate(
                    self.FRAMEWORK_NAME, self.pretrained_models, self.APP_OPTIONS
                )
                self._user_layout = self._gui.widgets
                self._user_layout_card = self._gui.card
            elif initialize_custom_gui_method.__func__ is not original_initialize_custom_gui_method:
                self._gui = GUI.ServingGUI()
                self._user_layout = self.initialize_custom_gui()
            else:
                initialize_custom_gui_method = getattr(self, "initialize_custom_gui", None)
                original_initialize_custom_gui_method = getattr(
                    Inference, "initialize_custom_gui", None
                )
                if (
                    initialize_custom_gui_method.__func__
                    is not original_initialize_custom_gui_method
                ):
                    self._gui = GUI.ServingGUI()
                    self._user_layout = self.initialize_custom_gui()
                else:
                    self.initialize_gui()

            def on_serve_callback(
                gui: Union[GUI.InferenceGUI, GUI.ServingGUI, GUI.ServingGUITemplate]
            ):
                Progress("Deploying model ...", 1)
                if isinstance(self.gui, GUI.ServingGUITemplate):
                    deploy_params = self.get_params_from_gui()
                    model_files = self._download_model_files(
                        deploy_params["model_source"], deploy_params["model_files"]
                    )
                    deploy_params["model_files"] = model_files
                    self._load_model_headless(**deploy_params)
                elif isinstance(self.gui, GUI.ServingGUI):
                    deploy_params = self.get_params_from_gui()
                    self._load_model(deploy_params)
                else:  # GUI.InferenceGUI
                    device = gui.get_device()
                    self.device = device
                    self.load_on_device(self._model_dir, device)
                    gui.show_deployed_model_info(self)

            def on_change_model_callback(
                gui: Union[GUI.InferenceGUI, GUI.ServingGUI, GUI.ServingGUITemplate]
            ):
                self.shutdown_model()
                if isinstance(self.gui, (GUI.ServingGUI, GUI.ServingGUITemplate)):
                    self._api_request_model_layout.unlock()
                    self._api_request_model_layout.hide()
                    self.update_gui(self._model_served)
                    self._user_layout_card.show()

            self.gui.on_change_model_callbacks.append(on_change_model_callback)
            self.gui.on_serve_callbacks.append(on_serve_callback)
            self._initialize_app_layout()

        self._inference_requests = {}
        max_workers = 1 if not multithread_inference else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.predict = self._check_serve_before_call(self.predict)
        self.predict_raw = self._check_serve_before_call(self.predict_raw)
        self.get_info = self._check_serve_before_call(self.get_info)

        self.cache = InferenceImageCache(
            maxsize=sly_env.smart_cache_size(),
            ttl=sly_env.smart_cache_ttl(),
            is_persistent=True,
            base_folder=sly_env.smart_cache_container_dir(),
            log_progress=True,
        )

    def get_batch_size(self):
        if self.max_batch_size is not None:
            return min(self.DEFAULT_BATCH_SIZE, self.max_batch_size)
        return self.DEFAULT_BATCH_SIZE

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

    def _load_json_file(self, file_path: str) -> dict:
        if isinstance(file_path, str):
            if sly_fs.file_exists(file_path) and sly_fs.get_file_ext(file_path) == ".json":
                return sly_json.load_json_file(file_path)
            else:
                raise ValueError("File not found or invalid file format.")
        else:
            raise ValueError("Invalid file. Please provide a valid '.json' file.")

    def _load_models_json_file(self, models: str) -> List[Dict[str, Any]]:
        """
        Loads dictionary from the provided file.
        """
        if isinstance(models, str):
            if sly_fs.file_exists(models) and sly_fs.get_file_ext(models) == ".json":
                models = sly_json.load_json_file(models)
            else:
                raise ValueError("File not found or invalid file format.")
        else:
            raise ValueError("Invalid file. Please provide a valid '.json' file.")

        if not isinstance(models, list):
            raise ValueError("models parameters must be a list of dicts")
        for item in models:
            if not isinstance(item, dict):
                raise ValueError(f"Each item in models must be a dict.")
            model_meta = item.get("meta")
            if model_meta is None:
                raise ValueError(
                    "Model metadata not found. Please update provided models parameter to include key 'meta'."
                )
            model_files = model_meta.get("model_files")
            if model_files is None:
                raise ValueError(
                    "Model files not found in model metadata. "
                    "Please update provided models oarameter to include key 'model_files' in 'meta' key."
                )
        return models

    def get_ui(self) -> Widget:
        if not self._use_gui:
            return None
        return self.gui.get_ui()

    def initialize_custom_gui(self) -> Widget:
        raise NotImplementedError("Have to be implemented in child class after inheritance")

    def update_gui(self, is_model_deployed: bool = True) -> None:
        if isinstance(self.gui, (GUI.ServingGUI, GUI.ServingGUITemplate)):
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
        if isinstance(self.gui, GUI.ServingGUITemplate):
            return self.gui.get_params_from_gui()
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

    def _initialize_app_layout(self):
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

        if isinstance(self.gui, GUI.ServingGUITemplate):
            self._app_layout = Container(
                [self._user_layout_card, self._api_request_model_layout, self.get_ui()], gap=5
            )
            return

        if hasattr(self, "_user_layout"):
            self._user_layout_card = Card(
                title="Select Model",
                description="Select the model to deploy and press the 'Serve' button.",
                content=self._user_layout,
                lock_message="Model is deployed. To change the model, stop the serving first.",
            )
        else:
            self._user_layout_card = Card(
                title="Select Model",
                description="Select the model to deploy and press the 'Serve' button.",
                content=self._gui,
                lock_message="Model is deployed. To change the model, stop the serving first.",
            )

        self._app_layout = Container(
            [self._user_layout_card, self._api_request_model_layout, self.get_ui()], gap=5
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

        if sly_fs.dir_exists(src_path) or sly_fs.file_exists(
            src_path
        ):  # only during debug, has no effect in production
            dst_path = os.path.abspath(src_path)
            logger.info(f"File {dst_path} found.")
        elif src_path.startswith("/"):  # folder from Team Files
            team_id = sly_env.team_id()

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

                file_info = self.api.file.get_info_by_path(sly_env.team_id(), src_path)
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
            if not sly_fs.dir_exists(os.path.dirname(dst_path)):
                sly_fs.mkdir(os.path.dirname(dst_path))

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

    def _checkpoints_cache_dir(self):
        return os.path.join(os.path.expanduser("~"), ".cache", "supervisely", "checkpoints")

    def _download_model_files(
        self, model_source: str, model_files: List[str], log_progress: bool = True
    ) -> dict:
        if model_source == ModelSource.PRETRAINED:
            return self._download_pretrained_model(model_files, log_progress)
        elif model_source == ModelSource.CUSTOM:
            return self._download_custom_model(model_files, log_progress)

    def _download_pretrained_model(self, model_files: dict, log_progress: bool = True):
        """
        Downloads the pretrained model data.
        """
        local_model_files = {}
        cache_dir = self._checkpoints_cache_dir()

        for file in model_files:
            file_url = model_files[file]
            file_name = sly_fs.get_file_name_with_ext(file_url)
            if file_url.startswith("http"):
                with urlopen(file_url) as f:
                    file_size = f.length
                    file_name = get_filename_from_headers(file_url)
                    if file_name is None:
                        file_name = file
                    file_path = os.path.join(self.model_dir, file_name)
                    cached_path = os.path.join(cache_dir, file_name)
                    if os.path.exists(cached_path):
                        local_model_files[file] = cached_path
                        logger.debug(f"Model: '{file_name}' was found in checkpoint cache")
                        continue
                    if os.path.exists(file_path):
                        local_model_files[file] = file_path
                        logger.debug(f"Model: '{file_name}' was found in model dir")
                        continue

                    if log_progress:
                        with self.gui.download_progress(
                            message=f"Downloading: '{file_name}'",
                            total=file_size,
                            unit="bytes",
                            unit_scale=True,
                        ) as download_pbar:
                            self.gui.download_progress.show()
                            sly_fs.download(
                                url=file_url, save_path=file_path, progress=download_pbar.update
                            )
                    else:
                        sly_fs.download(url=file_url, save_path=file_path)
                    local_model_files[file] = file_path
            else:
                local_model_files[file] = file_url

        if log_progress:
            self.gui.download_progress.hide()
        return local_model_files

    def _download_custom_model(self, model_files: dict, log_progress: bool = True):
        """
        Downloads the custom model data.
        """
        team_id = sly_env.team_id()
        local_model_files = {}
        for file in model_files:
            file_url = model_files[file]
            file_info = self.api.file.get_info_by_path(team_id, file_url)
            if file_info is None:
                if sly_fs.file_exists(file_url):
                    local_model_files[file] = file_url
                    continue
                else:
                    raise FileNotFoundError(f"File '{file_url}' not found in Team Files")
            file_size = file_info.sizeb
            file_name = os.path.basename(file_url)
            file_path = os.path.join(self.model_dir, file_name)
            if log_progress:
                with self.gui.download_progress(
                    message=f"Downloading: '{file_name}'",
                    total=file_size,
                    unit="bytes",
                    unit_scale=True,
                ) as download_pbar:
                    self.gui.download_progress.show()
                    self.api.file.download(
                        team_id, file_url, file_path, progress_cb=download_pbar.update
                    )
            else:
                self.api.file.download(team_id, file_url, file_path)
            local_model_files[file] = file_path
        if log_progress:
            self.gui.download_progress.hide()
        return local_model_files

    def _load_model(self, deploy_params: dict):
        self.model_source = deploy_params.get("model_source")
        self.device = deploy_params.get("device")
        self.runtime = deploy_params.get("runtime", RuntimeType.PYTORCH)
        self.model_precision = deploy_params.get("model_precision", ModelPrecision.FP32)
        self._hardware = get_hardware_info(self.device)
        self.load_model(**deploy_params)
        self._model_served = True
        self._deploy_params = deploy_params
        if self.gui is not None:
            self.update_gui(self._model_served)
            self.gui.show_deployed_model_info(self)

    def _load_model_headless(
        self,
        model_files: dict,
        model_source: str,
        model_info: dict,
        device: str,
        runtime: str,
        **kwargs,
    ):
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
            **kwargs,
        }
        if model_source == ModelSource.CUSTOM:
            self._set_model_meta_custom_model(model_info)
            self._set_checkpoint_info_custom_model(deploy_params)

        try:
            if is_production():
                self._add_workflow_input(model_source, model_files, model_info)
        except Exception as e:
            logger.warning(f"Failed to add input to the workflow: {repr(e)}")

        self._load_model(deploy_params)
        if self._model_meta is None:
            self._set_model_meta_from_classes()

    def _set_model_meta_custom_model(self, model_info: dict):
        model_meta = model_info.get("model_meta")
        if model_meta is None:
            return
        if isinstance(model_meta, dict):
            self._model_meta = ProjectMeta.from_json(model_meta)
        elif isinstance(model_meta, str):
            remote_artifacts_dir = model_info["artifacts_dir"]
            model_meta_url = os.path.join(remote_artifacts_dir, model_meta)
            model_meta_path = self.download(model_meta_url)
            model_meta = sly_json.load_json_file(model_meta_path)
            self._model_meta = ProjectMeta.from_json(model_meta)
        else:
            raise ValueError(
                "model_meta should be a dict or a name of '.json' file in experiment artifacts folder in Team Files"
            )
        self._get_confidence_tag_meta()
        self.classes = [obj_class.name for obj_class in self._model_meta.obj_classes]

    def _set_checkpoint_info_custom_model(self, deploy_params: dict):
        model_info = deploy_params.get("model_info", {})
        model_files = deploy_params.get("model_files", {})
        if model_info:
            checkpoint_name = os.path.basename(model_files.get("checkpoint"))
            checkpoint_file_path = os.path.join(
                model_info.get("artifacts_dir"), "checkpoints", checkpoint_name
            )
            checkpoint_file_info = self.api.file.get_info_by_path(
                sly_env.team_id(), checkpoint_file_path
            )
            if checkpoint_file_info is None:
                checkpoint_url = None
            else:
                checkpoint_url = self.api.file.get_url(checkpoint_file_info.id)

            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=checkpoint_name,
                model_name=model_info.get("model_name"),
                architecture=model_info.get("framework_name"),
                checkpoint_url=checkpoint_url,
                custom_checkpoint_path=checkpoint_file_path,
                model_source=ModelSource.CUSTOM,
            )

    def shutdown_model(self):
        self._model_served = False
        self.device = None
        self.runtime = None
        self.model_precision = None
        self.checkpoint_info = None
        self.max_batch_size = None
        clean_up_cuda()
        logger.info("Model has been stopped")

    def _on_model_deployed(self):
        pass

    def get_classes(self) -> List[str]:
        return self.classes

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
            "batch_inference_support": self.is_batch_inference_supported(),
            "max_batch_size": self.max_batch_size,
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

    def _get_deploy_info(self) -> DeployInfo:
        if self.checkpoint_info is None:
            raise ValueError("Checkpoint info is not set.")
        deploy_info = {
            **asdict(self.checkpoint_info),
            "task_type": self.get_info()["task type"],
            "device": self.device,
            "runtime": self.runtime,
            "model_precision": self.model_precision,
            "hardware": self._hardware,
            "deploy_params": self._deploy_params,
        }
        return DeployInfo(**deploy_info)

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

    def _set_model_meta_from_classes(self):
        classes = self.get_classes()
        if not classes:
            raise ValueError("Can't create model meta. Please, set the `self.classes` attribute.")
        shape = self._get_obj_class_shape()
        self._model_meta = ProjectMeta([ObjClass(name, shape) for name in classes])
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
        image_path: Union[str, np.ndarray],
        predictions: List[Prediction],
        classes_whitelist: Optional[List[str]] = None,
    ) -> Annotation:
        labels = []
        for prediction in predictions:
            if (
                not classes_whitelist in (None, "all")
                and hasattr(prediction, "class_name")
                and prediction.class_name not in classes_whitelist
            ):
                continue
            if "classes_whitelist" in inspect.signature(self._create_label).parameters:
                # pylint: disable=unexpected-keyword-arg
                # pylint: disable=too-many-function-args
                label = self._create_label(prediction, classes_whitelist)
            else:
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
            img = sly_image.read(image_path)
            img_size = img.shape[:2]
        else:
            img_size = image_path.shape[:2]
        ann = Annotation(img_size, labels)
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

    def _inference_auto(
        self,
        source: List[Union[str, np.ndarray]],
        settings: Dict[str, Any],
    ) -> Tuple[List[Annotation], List[dict]]:
        inference_mode = settings.get("inference_mode", "full_image")
        use_raw = (
            inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced"
        )
        is_predict_batch_raw_implemented = (
            type(self).predict_batch_raw != Inference.predict_batch_raw
        )
        if (not use_raw and self.is_batch_inference_supported()) or (
            use_raw and is_predict_batch_raw_implemented
        ):
            return self._inference_batched_wrapper(source, settings)
        else:
            return self._inference_one_by_one_wrapper(source, settings)

    def _inference_batched_wrapper(
        self,
        source: List[Union[str, np.ndarray]],
        settings: Dict,
    ) -> Tuple[List[Annotation], List[dict]]:
        # This method read images and collect slides_data (data_to_return)
        images_np = [sly_image.read(img) if isinstance(img, str) else img for img in source]
        slides_data = []
        anns = self._inference_batched(
            source=images_np,
            settings=settings,
            data_to_return=slides_data,
        )
        return anns, slides_data

    @process_images_batch_sliding_window
    @process_images_batch_roi
    def _inference_batched(
        self,
        source: List[np.ndarray],
        settings: Dict,
        data_to_return: list,
    ) -> List[Annotation]:
        images_np = source
        inference_mode = settings.get("inference_mode", "full_image")
        use_raw = (
            inference_mode == "sliding_window" and settings["sliding_window_mode"] == "advanced"
        )
        if not use_raw:
            predictions = self.predict_batch(images_np=images_np, settings=settings)
        else:
            predictions = self.predict_batch_raw(images_np=images_np, settings=settings)
        anns = []
        for src, prediction in zip(source, predictions):
            ann = self._predictions_to_annotation(
                src, prediction, classes_whitelist=settings.get("classes", None)
            )
            anns.append(ann)
        return anns

    def _inference_one_by_one_wrapper(
        self,
        source: List[Union[str, np.ndarray]],
        settings: Dict[str, Any],
    ) -> Tuple[List[Annotation], List[dict]]:
        anns = []
        slides_data = []
        writer = TempImageWriter()
        for img in source:
            if isinstance(img, np.ndarray):
                image_path = writer.write(img)
            else:
                image_path = img
            data_to_return = {}
            ann = self._inference_image_path(
                image_path=image_path,
                settings=settings,
                data_to_return=data_to_return,
            )
            anns.append(ann)
            slides_data.append(data_to_return)
        writer.clean()
        return anns, slides_data

    @process_image_sliding_window
    @process_image_roi
    def _inference_image_path(
        self,
        image_path: str,
        settings: Dict,
        data_to_return: Dict,  # for decorators
    ) -> Annotation:
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

    def _inference_benchmark(
        self,
        images_np: List[np.ndarray],
        settings: dict,
    ) -> Tuple[List[Annotation], dict]:
        t0 = time.time()
        predictions, benchmark = self.predict_benchmark(images_np, settings)
        total_time = (time.time() - t0) * 1000  # ms
        benchmark = {
            "total": total_time,
            "preprocess": benchmark.get("preprocess"),
            "inference": benchmark.get("inference"),
            "postprocess": benchmark.get("postprocess"),
        }
        anns = []
        for i, image_np in enumerate(images_np):
            ann = self._predictions_to_annotation(image_np, predictions[i])
            anns.append(ann)
        return anns, benchmark

    # pylint: disable=method-hidden
    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[Prediction]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def predict_batch(
        self, images_np: List[np.ndarray], settings: Dict[str, Any]
    ) -> List[List[Prediction]]:
        """Predict batch of images. `images_np` is a list of numpy arrays in RGB format

        If this method is not overridden in a subclass, the following fallback logic works:
            - If predict_benchmark is overridden, then call predict_benchmark
            - Otherwise, raise NotImplementedError
        """
        is_predict_benchmark_overridden = (
            type(self).predict_benchmark != Inference.predict_benchmark
        )
        if is_predict_benchmark_overridden:
            return self.predict_benchmark(images_np, settings)[0]
        else:
            raise NotImplementedError("Have to be implemented in child class")

    def predict_batch_raw(
        self, images_np: List[np.ndarray], settings: Dict[str, Any]
    ) -> List[List[Prediction]]:
        """Predict batch of images. `source` is a list of numpy arrays in RGB format"""
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def predict_benchmark(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[Prediction]], dict]:
        """
        Inference a batch of images with speedtest benchmarking.

        :param images_np: list of numpy arrays in RGB format
        :param settings: inference settings

        :return: tuple of annotation and benchmark dict with speedtest results in milliseconds.
            The benchmark dict should contain the following keys (all values in milliseconds):
            - preprocess: time of preprocessing (e.g. image loading, resizing, etc.)
            - inference: time of inference. Consider to include not only the time of the model forward pass, but also
                steps like NMS (Non-Maximum Suppression), decoding module, and everything that is done to calculate meaningful predictions.
            - postprocess: time of postprocessing (e.g. resizing output masks, aligning predictions with the input image, formatting, etc.)
            If some of the keys are missing, they will be considered as None.

        Note:
        If this method is not overridden in a subclass, the following fallback logic works:
            - If predict_batch is overridden, then call it
            - If predict_batch is not overridden but the batch size is 1, then use `predict`
            - If predict_batch is not overridden and the batch size is greater than 1, then raise NotImplementedError
        """
        is_predict_batch_overridden = type(self).predict_batch != Inference.predict_batch
        empty_benchmark = {}
        if is_predict_batch_overridden:
            predictions = self.predict_batch(images_np, settings)
            return predictions, empty_benchmark
        elif len(images_np) == 1:
            image_np = images_np[0]
            writer = TempImageWriter()
            image_path = writer.write(image_np)
            prediction = self.predict(image_path, settings)
            writer.clean()
            return [prediction], empty_benchmark
        else:
            raise NotImplementedError("Have to be implemented in child class")

    def is_batch_inference_supported(self) -> bool:
        is_predict_batch_overridden = type(self).predict_batch != Inference.predict_batch
        is_predict_benchmark_overridden = (
            type(self).predict_benchmark != Inference.predict_benchmark
        )
        return is_predict_batch_overridden or is_predict_benchmark_overridden

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

    def _format_output(
        self,
        anns: List[Annotation],
        slides_data: List[dict] = None,
    ) -> List[dict]:
        if not slides_data:
            slides_data = [{} for _ in range(len(anns))]
        assert len(anns) == len(slides_data)
        return [{"annotation": ann.to_json(), "data": data} for ann, data in zip(anns, slides_data)]

    def _inference_image(self, state: dict, file: UploadFile):
        logger.debug("Inferring image...", extra={"state": state})
        settings = self._get_inference_settings(state)
        image_np = sly_image.read_bytes(file.file.read())
        logger.debug("Inference settings:", extra=settings)
        logger.debug("Image info:", extra={"w": image_np.shape[1], "h": image_np.shape[0]})
        anns, slides_data = self._inference_auto(
            [image_np],
            settings=settings,
        )
        results = self._format_output(anns, slides_data)
        return results[0]

    def _inference_batch(self, state: dict, files: List[UploadFile]):
        logger.debug("Inferring batch...", extra={"state": state})
        settings = self._get_inference_settings(state)
        images = [sly_image.read_bytes(file.file.read()) for file in files]
        anns, slides_data = self._inference_auto(
            images,
            settings=settings,
        )
        return self._format_output(anns, slides_data)

    def _inference_batch_ids(self, api: Api, state: dict):
        logger.debug("Inferring batch_ids...", extra={"state": state})
        settings = self._get_inference_settings(state)
        ids = state["batch_ids"]
        infos = api.image.get_info_by_id_batch(ids)
        datasets = defaultdict(list)
        for info in infos:
            datasets[info.dataset_id].append(info.id)
        results = []
        for dataset_id, ids in datasets.items():
            images_np = api.image.download_nps(dataset_id, ids)
            anns, slides_data = self._inference_auto(
                source=images_np,
                settings=settings,
            )
            results.extend(self._format_output(anns, slides_data))
        return results

    def _inference_image_id(self, api: Api, state: dict, async_inference_request_uuid: str = None):
        logger.debug("Inferring image_id...", extra={"state": state})
        settings = self._get_inference_settings(state)
        upload = state.get("upload", False)
        image_id = state["image_id"]
        image_info = api.image.get_info_by_id(image_id)
        image_np = api.image.download_np(image_id)
        logger.debug("Inference settings:", extra=settings)
        logger.debug(
            "Image info:",
            extra={"id": image_id, "w": image_info.width, "h": image_info.height},
        )

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

        anns, slides_data = self._inference_auto(
            [image_np],
            settings=settings,
        )
        ann = anns[0]

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

        result = self._format_output(anns, slides_data)[0]
        if async_inference_request_uuid is not None and ann is not None:
            inference_request["result"] = result
        return result

    def _inference_image_url(self, api: Api, state: dict):
        logger.debug("Inferring image_url...", extra={"state": state})
        settings = self._get_inference_settings(state)
        image_url = state["image_url"]
        ext = sly_fs.get_file_ext(image_url)
        if ext == "":
            ext = ".jpg"
        image_path = os.path.join(get_data_dir(), rand_str(15) + ext)
        sly_fs.download(image_url, image_path)
        logger.debug("Inference settings:", extra=settings)
        logger.debug(f"Downloaded path: {image_path}")
        anns, slides_data = self._inference_auto(
            [image_path],
            settings=settings,
        )
        sly_fs.silent_remove(image_path)
        return self._format_output(anns, slides_data)[0]

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
        batch_size = state.get("batch_size", None)
        if batch_size is None:
            batch_size = self.get_batch_size()
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
            frames = self.cache.download_frames(api, video_info.id, batch, redownload_video=True)
            anns, slides_data = self._inference_auto(
                source=frames,
                settings=settings,
            )
            if tracker is not None:
                for frame_index, frame, ann in zip(batch, frames, anns):
                    tracks_data = tracker.update(frame, ann, frame_index, tracks_data)
            batch_results = self._format_output(anns, slides_data)
            results.extend(batch_results)
            if async_inference_request_uuid is not None:
                sly_progress.iters_done(len(batch))
                inference_request["pending_results"].extend(batch_results)
            logger.debug(f"Frames {batch[0]}-{batch[-1]} done.")
        video_ann_json = None
        if tracker is not None:
            video_ann_json = tracker.get_annotation(
                tracks_data, (video_info.frame_height, video_info.frame_width), n_frames
            ).to_json()
        result = {"ann": results, "video_ann": video_ann_json}
        if async_inference_request_uuid is not None and len(results) > 0:
            inference_request["result"] = result.copy()
        return result

    def _inference_images_ids(
        self,
        api: Api,
        state: dict,
        images_ids: List[int],
        async_inference_request_uuid: str = None,
    ):
        """Inference images by ids.
        If "output_project_id" in state, upload images and annotations to the output project.
        If "output_project_id" equal to source project id, upload annotations to the source project.
        If "output_project_id" is None, write annotations to inference request object.
        """
        logger.debug("Inferring images...", extra={"state": state})
        batch_size = state.get("batch_size", None)
        if batch_size is None:
            batch_size = self.get_batch_size()
        output_project_id = state.get("output_project_id", None)
        images_infos = api.image.get_info_by_id_batch(images_ids)
        images_infos_dict = {im_info.id: im_info for im_info in images_infos}
        dataset_infos_dict = {
            ds_id: api.dataset.get_info_by_id(ds_id)
            for ds_id in set([im_info.dataset_id for im_info in images_infos])
        }

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
            sly_progress.total = len(images_ids)

        def _download_images(images_ids):
            with ThreadPoolExecutor(max(8, min(batch_size, 64))) as executor:
                for image_id in images_ids:
                    executor.submit(
                        self.cache.download_image,
                        api,
                        image_id,
                    )

        # start downloading in parallel
        threading.Thread(target=_download_images, args=[images_ids], daemon=True).start()

        output_project_metas_dict = {}

        def _upload_results_to_source(results: List[Dict]):
            nonlocal output_project_metas_dict
            for result in results:
                image_id = result["image_id"]
                image_info: ImageInfo = images_infos_dict[image_id]
                dataset_info: DatasetInfo = dataset_infos_dict[image_info.dataset_id]
                project_id = dataset_info.project_id
                ann = Annotation.from_json(result["annotation"], self.model_meta)
                output_project_meta = output_project_metas_dict.get(project_id, None)
                if output_project_meta is None:
                    output_project_meta = ProjectMeta.from_json(
                        api.project.get_meta(output_project_id)
                    )
                output_project_meta, ann, meta_changed = update_meta_and_ann(
                    output_project_meta, ann
                )
                output_project_metas_dict[project_id] = output_project_meta
                if meta_changed:
                    output_project_meta = api.project.update_meta(project_id, output_project_meta)
                ann = update_classes(api, ann, output_project_meta, project_id)
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

        def _add_results_to_request(results: List[Dict]):
            if async_inference_request_uuid is None:
                return
            inference_request["pending_results"].extend(results)
            sly_progress.iters_done(len(results))

        new_dataset_id = {}

        def _get_or_create_new_dataset(output_project_id, src_dataset_id):
            """Copy dataset in output project if not exists and return its id"""
            if src_dataset_id in new_dataset_id:
                return new_dataset_id[src_dataset_id]
            dataset_info = api.dataset.get_info_by_id(src_dataset_id)
            output_dataset_id = api.dataset.create(
                output_project_id, dataset_info.name, change_name_if_conflict=True
            ).id
            new_dataset_id[src_dataset_id] = output_dataset_id
            return output_dataset_id

        def _copy_images_to_dst(
            src_dataset_id, dst_dataset_id, image_infos, dst_names
        ) -> List[ImageInfo]:
            return api.image.copy_batch_optimized(
                src_dataset_id,
                image_infos,
                dst_dataset_id,
                dst_names=dst_names,
                with_annotations=False,
                skip_validation=True,
            )

        def _upload_results_to_other(results: List[Dict]):
            nonlocal output_project_metas_dict
            if len(results) == 0:
                return
            src_dataset_id = results[0]["dataset_id"]
            dataset_id = _get_or_create_new_dataset(output_project_id, src_dataset_id)
            src_image_infos = [images_infos_dict[result["image_id"]] for result in results]
            image_names = [result["image_name"] for result in results]
            image_infos = _copy_images_to_dst(
                src_dataset_id, dataset_id, src_image_infos, image_names
            )
            image_infos.sort(key=lambda x: image_names.index(x.name))
            api.logger.debug(
                "Uploading results to other project...",
                extra={
                    "src_dataset_id": src_dataset_id,
                    "dst_project_id": output_project_id,
                    "dst_dataset_id": dataset_id,
                    "items_count": len(image_infos),
                },
            )
            meta_changed = False
            anns = []
            for result in results:
                ann = Annotation.from_json(result["annotation"], self.model_meta)
                output_project_meta = output_project_metas_dict.get(output_project_id, None)
                if output_project_meta is None:
                    output_project_meta = ProjectMeta.from_json(
                        api.project.get_meta(output_project_id)
                    )
                output_project_meta, ann, c = update_meta_and_ann(output_project_meta, ann)
                output_project_metas_dict[output_project_id] = output_project_meta
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

        def upload_results_to_source_or_other(results: List[Dict]):
            if len(results) == 0:
                return
            dataset_id = results[0]["dataset_id"]
            dataset_info: DatasetInfo = dataset_infos_dict[dataset_id]
            project_id = dataset_info.project_id
            if project_id == output_project_id:
                _upload_results_to_source(results)
            else:
                _upload_results_to_other(results)

        if output_project_id is None:
            upload_f = _add_results_to_request
        else:
            upload_f = upload_results_to_source_or_other

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
                            for each in batch:
                                ds_batches.setdefault(each["dataset_id"], []).append(each)
                        for _, joined_batch in ds_batches.items():
                            upload_f(joined_batch)
                        continue
                    if stop_event.is_set():
                        self._on_inference_end(None, async_inference_request_uuid)
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                raise

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
        stop = False
        try:
            for image_ids_batch in batched(images_ids, batch_size=batch_size):
                if stop:
                    break
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

                images_nps = [self.cache.download_image(api, img_id) for img_id in image_ids_batch]
                anns, slides_data = self._inference_auto(
                    source=images_nps,
                    settings=settings,
                )
                batch_results = []
                for i, ann in enumerate(anns):
                    image_info: ImageInfo = images_infos_dict[image_ids_batch[i]]
                    batch_results.append(
                        {
                            "annotation": ann.to_json(),
                            "data": slides_data[i],
                            "image_id": image_info.id,
                            "image_name": image_info.name,
                            "dataset_id": image_info.dataset_id,
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
        batch_size = state.get("batch_size", None)
        if batch_size is None:
            batch_size = self.get_batch_size()

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
                with ThreadPoolExecutor(max(8, min(batch_size, 64))) as executor:
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
            if dataset_info.parent_id is None:
                output_dataset_id = api.dataset.copy(
                    output_project_id,
                    src_dataset_id,
                    dataset_info.name,
                    change_name_if_conflict=True,
                ).id
            else:
                parent_dataset_id = _get_or_create_new_dataset(
                    output_project_id, dataset_info.parent_id
                )
                output_dataset_info = api.dataset.create(
                    output_project_id, dataset_info.name, parent_id=parent_dataset_id
                )
                api.image.copy_batch_optimized(
                    dataset_info.id,
                    images_infos_dict[dataset_info.id],
                    output_dataset_info.id,
                    with_annotations=False,
                )
                output_dataset_id = output_dataset_info.id
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
                dataset_id,
                filters=[{"field": "name", "operator": "in", "value": image_names}],
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
                        self._on_inference_end(None, async_inference_request_uuid)
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
                        images_paths, _ = zip(
                            *read_from_cached_project(
                                project_info.id,
                                dataset_info.name,
                                [ii.name for ii in images_infos_batch],
                            )
                        )
                        images_nps = [sly_image.read(img_path) for img_path in images_paths]
                    else:
                        images_nps = self.cache.download_images(
                            api,
                            dataset_info.id,
                            [info.id for info in images_infos_batch],
                            return_images=True,
                        )
                    anns, slides_data = self._inference_auto(
                        source=images_nps,
                        settings=settings,
                    )
                    batch_results = []
                    for i, ann in enumerate(anns):
                        batch_results.append(
                            {
                                "annotation": ann.to_json(),
                                "data": slides_data[i],
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

    def _run_speedtest(
        self,
        api: Api,
        state: dict,
        async_inference_request_uuid: str = None,
    ):
        """Run speedtest on project images."""
        logger.debug("Running speedtest...", extra={"state": state})
        project_id = state["projectId"]
        batch_size = state["batch_size"]
        num_iterations = state["num_iterations"]
        num_warmup = state.get("num_warmup", 3)
        dataset_ids = state.get("dataset_ids", None)
        cache_project_on_model = state.get("cache_project_on_model", False)
        max_images_number = 100

        datasets_infos = api.dataset.get_list(project_id, recursive=True)
        datasets_infos_dict = {ds_info.id: ds_info for ds_info in datasets_infos}
        if dataset_ids is not None:
            datasets_infos = [
                datasets_infos_dict[dataset_id]
                for dataset_id in dataset_ids
                if dataset_id in datasets_infos_dict
            ]

        # progress
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
            sly_progress.total = num_iterations
            sly_progress.current = 0

            preparing_progress = inference_request["preparing_progress"]

        preparing_progress["current"] = 0
        preparing_progress["total"] = len(datasets_infos)
        preparing_progress["status"] = "download_info"
        images_infos_dict = {}
        for dataset_info in datasets_infos:
            images_infos_dict[dataset_info.id] = api.image.get_list(dataset_info.id)
            if not cache_project_on_model:
                preparing_progress["current"] += 1

        if cache_project_on_model:

            def _progress_cb(count: int = 1):
                preparing_progress["current"] += count

            preparing_progress["current"] = 0
            preparing_progress["total"] = sum(
                dataset_info.items_count for dataset_info in datasets_infos
            )
            preparing_progress["status"] = "download_project"
            download_to_cache(api, project_id, datasets_infos, progress_cb=_progress_cb)

        preparing_progress["status"] = "warmup"
        preparing_progress["current"] = 0
        preparing_progress["total"] = num_warmup

        images_infos: List[ImageInfo] = [
            image_info for infos in images_infos_dict.values() for image_info in infos
        ][:max_images_number]

        def _download_images():
            with ThreadPoolExecutor(max(8, min(batch_size, 64))) as executor:
                for image_info in images_infos:
                    executor.submit(
                        self.cache.download_image,
                        api,
                        image_info.id,
                    )

        if not cache_project_on_model:
            # start downloading in parallel
            threading.Thread(target=_download_images, daemon=True).start()

        def _add_results_to_request(results: List[Dict]):
            if async_inference_request_uuid is None:
                return
            inference_request["pending_results"].append(results)
            sly_progress.iters_done(1)

        def _upload_loop(q: Queue, stop_event: threading.Event, api: Api, upload_f: Callable):
            try:
                while True:
                    items = []
                    while not q.empty():
                        items.append(q.get_nowait())
                    if len(items) > 0:
                        for batch in items:
                            upload_f(batch)
                        continue
                    if stop_event.is_set():
                        self._on_inference_end(None, async_inference_request_uuid)
                        return
                    time.sleep(1)
            except Exception as e:
                api.logger.error("Error in upload loop: %s", str(e), exc_info=True)
                raise

        upload_f = _add_results_to_request

        upload_queue = Queue()
        stop_upload_event = threading.Event()
        threading.Thread(
            target=_upload_loop,
            args=[upload_queue, stop_upload_event, api, upload_f],
            daemon=True,
        ).start()

        settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=settings)
        results = []
        stop = False

        def image_batch_generator(batch_size):
            logger.debug(
                f"image_batch_generator. images_infos={len(images_infos)}, batch_size={batch_size}"
            )
            batch = []
            while True:
                for image_info in images_infos:
                    batch.append(image_info)
                    if len(batch) == batch_size:
                        logger.debug("yield batch")
                        yield batch
                        batch = []

        batch_generator = image_batch_generator(batch_size)
        try:
            for i in range(num_iterations + num_warmup):
                if stop:
                    break
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
                if i == num_warmup:
                    preparing_progress["status"] = "inference"

                images_infos_batch: List[ImageInfo] = next(batch_generator)

                images_infos_batch_by_dataset = {}
                for image_info in images_infos_batch:
                    images_infos_batch_by_dataset.setdefault(image_info.dataset_id, []).append(
                        image_info
                    )

                # Read images
                if cache_project_on_model:
                    images_nps = []
                    for (
                        dataset_id,
                        images_infos,
                    ) in images_infos_batch_by_dataset.items():
                        dataset_info = datasets_infos_dict[dataset_id]
                        images_paths, _ = zip(
                            *read_from_cached_project(
                                project_id,
                                dataset_info.name,
                                [ii.name for ii in images_infos],
                            )
                        )
                        images_nps.extend([sly_image.read(path) for path in images_paths])
                else:
                    images_nps = []
                    for (
                        dataset_id,
                        images_infos,
                    ) in images_infos_batch_by_dataset.items():
                        images_nps.extend(
                            self.cache.download_images(
                                api,
                                dataset_id,
                                [info.id for info in images_infos],
                                return_images=True,
                            )
                        )
                # Inference
                anns, benchmark = self._inference_benchmark(
                    images_np=images_nps,
                    settings=settings,
                )
                # Collect results if warmup is done
                if i >= num_warmup:
                    results.append(benchmark)
                    upload_queue.put(benchmark)
                else:
                    preparing_progress["current"] += 1
        except Exception:
            stop_upload_event.set()
            raise
        if async_inference_request_uuid is not None and len(results) > 0:
            inference_request["result"] = results
        stop_upload_event.set()
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

    def is_model_deployed(self):
        return self._model_served

    def schedule_task(self, func, *args, **kwargs):
        inference_request_uuid = kwargs.get("inference_request_uuid", None)
        if inference_request_uuid is None:
            self._executor.submit(func, *args, **kwargs)
        else:
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                func,
                *args,
                **kwargs,
            )
            end_callback = partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
        logger.debug("Scheduled task.", extra={"inference_request_uuid": inference_request_uuid})

    def serve(self):
        if not self._use_gui:
            Progress("Deploying model ...", 1)

        if is_debug_with_sly_net():
            # advanced debug for Supervisely Team
            logger.warning(
                "Serving is running in advanced development mode with Supervisely VPN Network"
            )
            team_id = sly_env.team_id()
            # sly_app_development.supervisely_vpn_network(action="down") # for debug
            sly_app_development.supervisely_vpn_network(action="up")
            task = sly_app_development.create_debug_task(team_id, port="8000")
            self._task_id = task["id"]
            os.environ["TASK_ID"] = str(self._task_id)
        else:
            if not self._is_local_deploy:
                self._task_id = sly_env.task_id() if is_production() else None

        if isinstance(self.gui, GUI.InferenceGUI):
            self._app = Application(layout=self.get_ui())
        elif isinstance(self.gui, GUI.ServingGUI):
            self._app = Application(layout=self._app_layout)
        # elif isinstance(self.gui, GUI.InferenceGUI):
        #     self._app = Application(layout=self.get_ui())
        else:
            self._app = Application(layout=self.get_ui())

        server = self._app.get_server()
        self._app.set_ready_check_function(self.is_model_deployed)

        if self._is_local_deploy:
            # Predict and shutdown
            if any(
                [
                    self._args.predict_project,
                    self._args.predict_dataset,
                    self._args.predict_dir,
                    self._args.predict_image,
                ]
            ):
                self._inference_by_local_deploy_args()
                # Gracefully shut down the server
                self._app.shutdown()
        # else: run server after endpoints

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

        self.cache.add_cache_endpoint(server)
        self.cache.add_cache_files_endpoint(server)

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
        def inference_batch_ids(response: Response, request: Request):
            # check batch size
            batch_size = len(request.state.state["batch_ids"])
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
            logger.debug(f"'inference_batch_ids' request in json format:{request.state.state}")
            return self._inference_batch_ids(request.state.api, request.state.state)

        @server.post("/inference_batch_ids_async")
        def inference_batch_ids_async(response: Response, request: Request):
            logger.debug(
                f"'inference_batch_ids_async' request in json format:{request.state.state}"
            )
            images_ids = request.state.state["images_ids"]
            # check batch size
            batch_size = request.state.state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._inference_images_ids,
                request.state.api,
                request.state.state,
                images_ids,
                inference_request_uuid,
            )
            end_callback = partial(
                self._on_inference_end, inference_request_uuid=inference_request_uuid
            )
            future.add_done_callback(end_callback)
            logger.debug(
                "Inference has scheduled from 'inference_batch_ids_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post("/inference_video_id")
        def inference_video_id(response: Response, request: Request):
            logger.debug(f"'inference_video_id' request in json format:{request.state.state}")
            # check batch size
            batch_size = request.state.state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
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
                # check batch size
                batch_size = len(files)
                if self.max_batch_size is not None and batch_size > self.max_batch_size:
                    response.status_code = status.HTTP_400_BAD_REQUEST
                    return {
                        "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                        "success": False,
                    }
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
        def inference_video_id_async(response: Response, request: Request):
            logger.debug(f"'inference_video_id_async' request in json format:{request.state.state}")
            # check batch size
            batch_size = request.state.state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
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
        def inference_project_id_async(response: Response, request: Request):
            logger.debug(
                f"'inference_project_id_async' request in json format:{request.state.state}"
            )
            project_id = request.state.state["projectId"]
            project_info = request.state.api.project.get_info_by_id(project_id)
            if project_info.type != str(ProjectType.IMAGES):
                raise ValueError("Only images projects are supported.")
            # check batch size
            batch_size = request.state.state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
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
            logger.debug(
                "Inference has scheduled from 'inference_project_id_async' endpoint",
                extra={"inference_request_uuid": inference_request_uuid},
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request_uuid,
            }

        @server.post("/run_speedtest")
        def run_speedtest(response: Response, request: Request):
            logger.debug(f"'run_speedtest' request in json format:{request.state.state}")
            project_id = request.state.state["projectId"]
            project_info = request.state.api.project.get_info_by_id(project_id)
            if project_info.type != str(ProjectType.IMAGES):
                response.status_code = status.HTTP_400_BAD_REQUEST
                response.body = {"message": "Only images projects are supported."}
                raise ValueError("Only images projects are supported.")
            batch_size = request.state.state["batch_size"]
            if batch_size > 1 and not self.is_batch_inference_supported():
                response.status_code = status.HTTP_501_NOT_IMPLEMENTED
                return {
                    "message": "Batch inference is not implemented for this model.",
                    "success": False,
                }
            # check batch size
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                    "success": False,
                }
            inference_request_uuid = uuid.uuid5(
                namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
            ).hex
            self._on_inference_start(inference_request_uuid)
            future = self._executor.submit(
                self._handle_error_in_async,
                inference_request_uuid,
                self._run_speedtest,
                request.state.api,
                request.state.state,
                inference_request_uuid,
            )
            logger.debug(
                "Speedtest has scheduled from 'run_speedtest' endpoint",
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
            inference_request.pop("lock", None)
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
                if isinstance(self.gui, GUI.ServingGUITemplate):
                    model_files = self._download_model_files(
                        deploy_params["model_source"], deploy_params["model_files"]
                    )
                    deploy_params["model_files"] = model_files
                    self._load_model_headless(**deploy_params)
                elif isinstance(self.gui, GUI.ServingGUI):
                    self._load_model(deploy_params)

                self.set_params_to_gui(deploy_params)
                # update to set correct device
                device = deploy_params.get("device", "cpu")
                self.gui.set_deployed(device)
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

        @server.post("/get_deploy_info")
        @self._check_serve_before_call
        def _get_deploy_info():
            return asdict(self._get_deploy_info())

        # Local deploy without predict args
        if self._is_local_deploy:
            self._run_server()

    def _parse_local_deploy_args(self):
        parser = argparse.ArgumentParser(description="Run Inference Serving")

        # Deploy args
        parser.add_argument(
            "--model",
            type=str,
            help="Name of the pretrained model or path to custom checkpoint file",
        )
        parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            default="cuda:0",
            help="Device to use for inference (default: 'cuda:0')",
        )
        parser.add_argument(
            "--runtime",
            type=str,
            choices=[RuntimeType.PYTORCH, RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT],
            default=RuntimeType.PYTORCH,
            help="Runtime type for inference (default: PYTORCH)",
        )
        # -------------------------- #

        # Predict args
        parser.add_argument("--predict-project", type=int, required=False, help="ID of the project")
        parser.add_argument(
            "--predict-dataset",
            type=lambda x: [int(i) for i in x.split(",")] if "," in x else int(x),
            required=False,
            help="ID of the dataset or a comma-separated list of dataset IDs",
        )
        parser.add_argument(
            "--predict-dir",
            type=str,
            required=False,
            help="Not implemented yet. Path to the local directory with images",
        )
        parser.add_argument(
            "--predict-image",
            type=str,
            required=False,
            help="Image ID on Supervisely instance or path to local image",
        )
        # -------------------------- #

        # Output args
        parser.add_argument("--output", type=str, required=False, help="Not implemented yet")
        parser.add_argument("--output-dir", type=str, required=False, help="Not implemented yet")
        # -------------------------- #

        # Parse arguments
        args, _ = parser.parse_known_args()
        if args.model is None:
            # raise ValueError("Argument '--model' is required for local deployment")
            return None, False
        if isinstance(args.predict_dataset, int):
            args.predict_dataset = [args.predict_dataset]
        if args.predict_image is not None:
            if args.predict_image.isdigit():
                args.predict_image = int(args.predict_image)
        return args, True

    def _get_pretrained_model_params_from_args(self):
        model_files = None
        model_source = None
        model_info = None
        need_download = True

        model = self._args.model
        for m in self.pretrained_models:
            meta = m.get("meta", None)
            if meta is None:
                continue
            model_name = meta.get("model_name", None)
            if model_name is None:
                continue
            m_files = meta.get("model_files", None)
            if m_files is None:
                continue
            checkpoint = m_files.get("checkpoint", None)
            if checkpoint is None:
                continue
            if model == m["meta"]["model_name"]:
                model_info = m
                model_source = ModelSource.PRETRAINED
                model_files = {"checkpoint": checkpoint}
                config = m_files.get("config", None)
                if config is not None:
                    model_files["config"] = config
                break

        return model_files, model_source, model_info, need_download

    def _get_custom_model_params_from_args(self):
        def _load_experiment_info(artifacts_dir):
            experiment_path = os.path.join(artifacts_dir, "experiment_info.json")
            model_info = self._load_json_file(experiment_path)
            original_model_files = model_info.get("model_files")
            if not original_model_files:
                raise ValueError("Invalid 'experiment_info.json'. Missing 'model_files' key.")
            return model_info, original_model_files

        def _prepare_local_model_files(artifacts_dir, checkpoint_path, original_model_files):
            return {k: os.path.join(artifacts_dir, v) for k, v in original_model_files.items()} | {
                "checkpoint": checkpoint_path
            }

        def _download_remote_files(team_id, artifacts_dir, local_artifacts_dir):
            sly_fs.mkdir(local_artifacts_dir, True)
            file_infos = self.api.file.list(team_id, artifacts_dir, False, "fileinfo")
            remote_paths = [f.path for f in file_infos if not f.is_dir]
            local_paths = [
                os.path.join(local_artifacts_dir, f.name) for f in file_infos if not f.is_dir
            ]

            coro = self.api.file.download_bulk_async(team_id, remote_paths, local_paths)
            loop = get_or_create_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                future.result()
            else:
                loop.run_until_complete(coro)

        model_source = ModelSource.CUSTOM
        need_download = False
        checkpoint_path = self._args.model

        if not os.path.isfile(checkpoint_path):
            team_id = sly_env.team_id(raise_not_found=False)
            if not team_id:
                raise ValueError(
                    "Team ID not found in env. Required for remote custom checkpoints."
                )
            file_info = self.api.file.get_info_by_path(team_id, checkpoint_path)
            if not file_info:
                raise ValueError(
                    f"Couldn't find: '{checkpoint_path}' locally or remotely in Team ID."
                )
            need_download = True

        artifacts_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        if not need_download:
            model_info, original_model_files = _load_experiment_info(artifacts_dir)
            model_files = _prepare_local_model_files(
                artifacts_dir, checkpoint_path, original_model_files
            )
        else:
            local_artifacts_dir = os.path.join(
                self.model_dir, "local_deploy", os.path.basename(artifacts_dir)
            )
            _download_remote_files(team_id, artifacts_dir, local_artifacts_dir)

            model_info, original_model_files = _load_experiment_info(local_artifacts_dir)
            model_files = _prepare_local_model_files(
                local_artifacts_dir, checkpoint_path, original_model_files
            )
        return model_files, model_source, model_info, need_download

    def _get_deploy_params_from_args(self):
        # Ensure model directory exists
        device = self._args.device if self._args.device else "cuda:0"
        runtime = self._args.runtime if self._args.runtime else RuntimeType.PYTORCH

        model_files, model_source, model_info, need_download = (
            self._get_pretrained_model_params_from_args()
        )
        if model_source is None:
            model_files, model_source, model_info, need_download = (
                self._get_custom_model_params_from_args()
            )

        if model_source is None:
            raise ValueError("Couldn't create 'model_source' from args")
        if model_files is None:
            raise ValueError("Couldn't create 'model_files' from args")
        if model_info is None:
            raise ValueError("Couldn't create 'model_info' from args")

        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
        }

        logger.info(f"Deploy parameters: {deploy_params}")
        return deploy_params, need_download

    def _run_server(self):
        config = uvicorn.Config(app=self._app, host="0.0.0.0", port=8000, ws="websockets")
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()

    def _inference_by_local_deploy_args(self):
        def predict_project_by_args(api: Api, project_id: int, dataset_ids: List[int] = None):
            source_project = api.project.get_info_by_id(project_id)
            workspace_id = source_project.workspace_id
            output_project = api.project.create(
                workspace_id, f"{source_project.name} predicted", change_name_if_conflict=True
            )
            results = self._inference_project_id(
                api=self.api,
                state={
                    "projectId": project_id,
                    "dataset_ids": dataset_ids,
                    "output_project_id": output_project.id,
                },
            )

        def predict_datasets_by_args(api: Api, dataset_ids: List[int]):
            dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
            project_ids = list(set([dataset_info.project_id for dataset_info in dataset_infos]))
            if len(project_ids) > 1:
                raise ValueError("All datasets should belong to the same project")
            predict_project_by_args(api, project_ids[0], dataset_ids)

        def predict_image_by_args(api: Api, image: Union[str, int]):
            def predict_image_np(image_np):
                settings = self._get_inference_settings({})
                anns, _ = self._inference_auto([image_np], settings)
                if len(anns) == 0:
                    return Annotation(img_size=image_np.shape[:2])
                ann = anns[0]
                return ann

            if isinstance(image, int):
                image_np = api.image.download_np(image)
                ann = predict_image_np(image_np)
                api.annotation.upload_ann(image, ann)
            elif isinstance(image, str):
                if sly_fs.file_exists(self._args.predict):
                    image_np = sly_image.read(self._args.predict)
                    ann = predict_image_np(image_np)
                    pred_ann_path = image + ".json"
                    sly_json.dump_json_file(ann.to_json(), pred_ann_path)
                    # Save image for debug
                    # ann.draw_pretty(image_np)
                    # pred_path = os.path.join(os.path.dirname(self._args.predict), "pred_" + os.path.basename(self._args.predict))
                    # sly_image.write(pred_path, image_np)

        if self._args.predict_project is not None:
            predict_project_by_args(self.api, self._args.predict_project)
        elif self._args.predict_dataset is not None:
            predict_datasets_by_args(self.api, self._args.predict_dataset)
        elif self._args.predict_dir is not None:
            raise NotImplementedError("Predict from directory is not implemented yet")
        elif self._args.predict_image is not None:
            predict_image_by_args(self.api, self._args.predict_image)

    def _add_workflow_input(self, model_source: str, model_files: dict, model_info: dict):
        if model_source == ModelSource.PRETRAINED:
            checkpoint_url = model_info["meta"]["model_files"]["checkpoint"]
            checkpoint_name = model_info["meta"]["model_name"]
        else:
            checkpoint_name = sly_fs.get_file_name_with_ext(model_files["checkpoint"])
            checkpoint_url = os.path.join(
                model_info["artifacts_dir"], "checkpoints", checkpoint_name
            )

        app_name = sly_env.app_name()
        meta = WorkflowMeta(node_settings=WorkflowSettings(title=f"Serve {app_name}"))

        logger.debug(
            f"Workflow Input: Checkpoint URL - {checkpoint_url}, Checkpoint Name - {checkpoint_name}"
        )
        if checkpoint_url and self.api.file.exists(sly_env.team_id(), checkpoint_url):
            self.api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
        else:
            logger.debug(
                f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input"
            )


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


def _fix_classes_names(meta: ProjectMeta, ann: Annotation):
    def _replace_strip(s, chars: str, replacement: str = "_") -> str:
        replace_pattern = f"^[{re.escape(chars)}]+|[{re.escape(chars)}]+$"
        return re.sub(replace_pattern, replacement, s)

    replaced_classes_in_meta = []
    for obj_class in meta.obj_classes:
        obj_class_name = _replace_strip(obj_class.name, " ", "")
        if obj_class_name != obj_class.name:
            new_obj_class = obj_class.clone(name=obj_class_name)
            meta = meta.delete_obj_class(obj_class.name)
            meta = meta.add_obj_class(new_obj_class)
            replaced_classes_in_meta.append((obj_class.name, obj_class_name))
    replaced_classes_in_ann = set()
    new_labels = []
    for label in ann.labels:
        obj_class = label.obj_class
        obj_class_name = _replace_strip(obj_class.name, " ", "")
        if obj_class_name != obj_class.name:
            new_obj_class = obj_class.clone(name=obj_class_name)
            label = label.clone(obj_class=new_obj_class)
            replaced_classes_in_ann.add((obj_class.name, obj_class_name))
        new_labels.append(label)
    ann = ann.clone(labels=new_labels)
    return meta, ann, replaced_classes_in_meta, list(replaced_classes_in_ann)


def update_meta_and_ann(meta: ProjectMeta, ann: Annotation):
    """Update project meta and annotation to match each other
    If obj class or tag meta from annotation conflicts with project meta
    add suffix to obj class or tag meta.
    Return tuple of updated project meta, annotation and boolean flag if meta was changed.
    """
    obj_classes_suffixes = ["_nn"]
    tag_meta_suffixes = ["_nn"]
    ann_obj_classes = {}
    ann_tag_metas = {}
    meta_changed = False

    meta, ann, replaced_classes_in_meta, replaced_classes_in_ann = _fix_classes_names(meta, ann)
    if replaced_classes_in_meta:
        meta_changed = True
        logger.warning(
            "Some classes names were fixed in project meta",
            extra={"replaced_classes": {old: new for old, new in replaced_classes_in_meta}},
        )

    # get all obj classes and tag metas from annotation
    for label in ann.labels:
        ann_obj_classes[label.obj_class.name] = label.obj_class
        for tag in label.tags:
            ann_tag_metas[tag.meta.name] = tag.meta
    for tag in ann.img_tags:
        ann_tag_metas[tag.meta.name] = tag.meta

    # check if obj classes are in project meta
    # if not, add them.
    # if shape is different, add them with suffix
    changed_obj_classes = {}
    for ann_obj_class in ann_obj_classes.values():
        if meta.get_obj_class(ann_obj_class.name) is None:
            meta = meta.add_obj_class(ann_obj_class)
            meta_changed = True
        elif (
            meta.get_obj_class(ann_obj_class.name).geometry_type != ann_obj_class.geometry_type
            and meta.get_obj_class(ann_obj_class.name).geometry_type != AnyGeometry
        ):
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


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = (self.end - self.start) * 1000  # ms

    def get_time(self):
        return self.duration


class TempImageWriter:
    def __init__(self, format: str = "png"):
        self.format = format
        self.temp_dir = os.path.join(get_data_dir(), rand_str(10))
        sly_fs.mkdir(self.temp_dir)

    def write(self, image: np.ndarray):
        image_path = os.path.join(self.temp_dir, f"{rand_str(10)}.{self.format}")
        sly_image.write(image_path, image)
        return image_path

    def clean(self):
        sly_fs.remove_dir(self.temp_dir)


def get_hardware_info(device: str) -> str:
    import platform

    device = device.lower()
    try:
        if device == "cpu":
            system = platform.system()
            if system == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif system == "Darwin":  # macOS
                command = "/usr/sbin/sysctl -n machdep.cpu.brand_string"
                return subprocess.check_output(command, shell=True).strip().decode()
            elif system == "Windows":
                command = "wmic cpu get name"
                output = subprocess.check_output(command, shell=True).decode()
                return output.strip().split("\n")[1].strip()
        elif "cuda" in device:
            idx = 0
            if ":" in device:
                idx = int(device.split(":")[1])
            gpus = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
                )
                .decode("utf-8")
                .strip()
            )
            gpu_list = gpus.split("\n")
            if idx >= len(gpu_list):
                raise ValueError(f"No GPU found at index {idx}")
            return gpu_list[idx]
    except Exception as e:
        logger.error("Error while getting hardware info", exc_info=True)
    return "Unknown"
