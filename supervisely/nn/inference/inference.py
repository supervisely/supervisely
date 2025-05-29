from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
import requests
import uvicorn
import yaml
from fastapi import Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from requests.structures import CaseInsensitiveDict
from tqdm import tqdm

import supervisely.app.development as sly_app_development
import supervisely.imaging.image as sly_image
import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
import supervisely.nn.inference.gui as GUI
from supervisely import DatasetInfo, batched
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
from supervisely.api.api import Api, ApiField
from supervisely.api.app_api import WorkflowMeta, WorkflowSettings
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import get_data_dir
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
from supervisely.io.fs import list_files
from supervisely.nn.inference.cache import InferenceImageCache
from supervisely.nn.inference.inference_request import (
    InferenceRequest,
    InferenceRequestsManager,
)
from supervisely.nn.inference.uploader import Uploader
from supervisely.nn.model.model_api import Prediction
from supervisely.nn.prediction_dto import Prediction as PredictionDTO
from supervisely.nn.utils import (
    CheckpointInfo,
    DeployInfo,
    ModelPrecision,
    ModelSource,
    RuntimeType,
    _get_model_name,
    get_gpu_usage,
    get_ram_usage,
)
from supervisely.project import ProjectType
from supervisely.project.download import download_to_cache, read_from_cached_project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress
from supervisely.video.video import ALLOWED_VIDEO_EXTENSIONS, VideoFrameReader

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


@dataclass
class AutoRestartInfo:
    deploy_params: dict

    class Fields:
        AUTO_RESTART_INFO = "autoRestartInfo"
        DEPLOY_PARAMS = "deployParams"

    def generate_fields(self) -> List[dict]:
        return [
            {
                ApiField.FIELD: self.Fields.AUTO_RESTART_INFO,
                ApiField.PAYLOAD: {self.Fields.DEPLOY_PARAMS: self.deploy_params},
            }
        ]

    @classmethod
    def from_response(cls, data: dict):
        autorestart_info = data.get(cls.Fields.AUTO_RESTART_INFO, None)
        if autorestart_info is None:
            return None
        return cls(deploy_params=autorestart_info.get(cls.Fields.DEPLOY_PARAMS, None))

    def is_changed(self, deploy_params: dict) -> bool:
        return self.deploy_params != deploy_params


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

        self.pretrained_models = self._load_models_json_file(self.MODELS) if self.MODELS else None
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

        self.autorestart = None
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
            self._use_gui = False
            deploy_params, need_download = self._get_deploy_params_from_args()
            if need_download:
                local_model_files = self._download_model_files(deploy_params, False)
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
                gui: Union[GUI.InferenceGUI, GUI.ServingGUI, GUI.ServingGUITemplate],
            ):
                Progress("Deploying model ...", 1)
                if isinstance(self.gui, GUI.ServingGUITemplate):
                    deploy_params = self.get_params_from_gui()
                    model_files = self._download_model_files(deploy_params)
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
                gui: Union[GUI.InferenceGUI, GUI.ServingGUI, GUI.ServingGUITemplate],
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

        self.inference_requests_manager = InferenceRequestsManager(executor=self._executor)

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
        """
        Set params for load_model method to GUI.
        """
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
                [self._user_layout_card, self._api_request_model_layout, self.get_ui()],
                gap=5,
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
            [self._user_layout_card, self._api_request_model_layout, self.get_ui()],
            gap=5,
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

    def _build_deploy_params_from_api(self, model_name: str, deploy_params: dict = None) -> dict:
        if deploy_params is None:
            deploy_params = {}
        selected_model = None
        for model in self.pretrained_models:
            if model["meta"]["model_name"].lower() == model_name.lower():
                selected_model = model
                break
        if selected_model is None:
            raise ValueError(f"Model {model_name} not found in models.json of serving app")
        deploy_params["model_files"] = selected_model["meta"]["model_files"]
        deploy_params["model_info"] = selected_model
        return deploy_params

    def _build_legacy_deploy_params_from_api(self, model_name: str) -> dict:
        selected_model = None
        if hasattr(self, "pretrained_models_table"):
            selected_model = self.pretrained_models_table.get_by_model_name(model_name)
            if selected_model is None:
                # @TODO: Improve error message
                raise ValueError("This app doesn't support new deploy api")

            self.pretrained_models_table.set_by_model_name(model_name)
            deploy_params = self.pretrained_models_table.get_selected_model_params()
            return deploy_params

    # @TODO: method name should be better?
    def _set_common_deploy_params(self, deploy_params: dict) -> dict:
        load_model_params = inspect.signature(self.load_model).parameters
        has_runtime_param = "runtime" in load_model_params

        if has_runtime_param:
            if deploy_params.get("runtime", None) is None:
                deploy_params["runtime"] = RuntimeType.PYTORCH
        if deploy_params.get("device", None) is None:
            deploy_params["device"] = "cuda:0" if get_gpu_count() > 0 else "cpu"
        return deploy_params

    def _download_model_files(self, deploy_params: dict, log_progress: bool = True) -> dict:
        if deploy_params["model_source"] == ModelSource.PRETRAINED:
            headless = self.gui is None
            return self._download_pretrained_model(
                deploy_params["model_files"], log_progress, headless
            )
        elif deploy_params["model_source"] == ModelSource.CUSTOM:
            if deploy_params["runtime"] != RuntimeType.PYTORCH:
                export = deploy_params["model_info"].get("export", {})
                if export is None:
                    export = {}
                export_model = export.get(deploy_params["runtime"], None)
                if export_model is not None:
                    if sly_fs.get_file_name(export_model) == sly_fs.get_file_name(
                        deploy_params["model_files"]["checkpoint"]
                    ):
                        deploy_params["model_files"]["checkpoint"] = (
                            deploy_params["model_info"]["artifacts_dir"] + export_model
                        )
                        logger.info(f"Found model checkpoint for '{deploy_params['runtime']}'")
            return self._download_custom_model(deploy_params["model_files"], log_progress)

    def _download_pretrained_model(
        self, model_files: dict, log_progress: bool = True, headless: bool = False
    ):
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
                        if not headless:
                            with self.gui.download_progress(
                                message=f"Downloading: '{file_name}'",
                                total=file_size,
                                unit="bytes",
                                unit_scale=True,
                            ) as download_pbar:
                                self.gui.download_progress.show()
                                sly_fs.download(
                                    url=file_url,
                                    save_path=file_path,
                                    progress=download_pbar.update,
                                )
                        else:
                            with tqdm(
                                total=file_size,
                                unit="bytes",
                                unit_scale=True,
                            ) as download_pbar:
                                logger.info(f"Downloading: '{file_name}'")
                                sly_fs.download(
                                    url=file_url, save_path=file_path, progress=download_pbar.update
                                )
                    else:
                        logger.info(f"Downloading: '{file_name}'")
                        sly_fs.download(url=file_url, save_path=file_path)
                    local_model_files[file] = file_path
            else:
                local_model_files[file] = file_url

        if log_progress:
            if self.gui is not None:
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
        if self._task_id is not None and is_production():
            try:
                if self.autorestart is None:
                    self.autorestart = AutoRestartInfo(self._deploy_params)
                    self.api.task.set_fields(self._task_id, self.autorestart.generate_fields())
                    logger.debug(
                        "Created new autorestart info.",
                        extra=self.autorestart.deploy_params,
                    )
                elif self.autorestart.is_changed(self._deploy_params):
                    self.autorestart.deploy_params.update(self._deploy_params)
                    self.api.task.set_fields(self._task_id, self.autorestart.generate_fields())
                    logger.debug(
                        "Autorestart info is changed. Parameters have been updated.",
                        extra=self.autorestart.deploy_params,
                    )
            except Exception as e:
                logger.warning(f"Failed to update autorestart info: {repr(e)}")
        if self.gui is not None:
            self.update_gui(self._model_served)
            self.gui.show_deployed_model_info(self)

    def load_custom_checkpoint(
        self, model_files: dict, model_meta: dict, device: Optional[str] = None, **kwargs
    ):
        """
        Loads local custom model checkpoint.

        :param: model_files: dict with local paths to model files
        :type: model_files: dict
        :param: model_meta: dict with model meta
        :type: model_meta: dict
        :param: device: device to load model on
        :type: device: str
        :param: kwargs: additional parameters will be passed to load_model method.
        :type: kwargs: dict
        :return: None
        :rtype: None

        :Usage Example:

         .. code-block:: python

            model_files = {
                "checkpoint": "supervisely_integration/serve/best.pth",
                "config": "supervisely_integration/serve/model_config.yml",
            }
            model_meta = sly.json.load_json_file("model_meta.json")

            model.load_custom_checkpoint(model_files, model_meta)
        """

        checkpoint = model_files.get("checkpoint")
        if checkpoint is None:
            raise ValueError("Model checkpoint is not provided")
        checkpoint_name = sly_fs.get_file_name_with_ext(model_files["checkpoint"])

        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=checkpoint_name,
            model_name=None,
            architecture=None,
            checkpoint_url=None,
            custom_checkpoint_path=None,
            model_source=ModelSource.CUSTOM,
        )

        deploy_params = {
            "model_source": ModelSource.CUSTOM,
            "model_files": model_files,
            "model_info": {},
            "device": device,
            "runtime": RuntimeType.PYTORCH,
        }
        deploy_params.update(kwargs)

        # TODO: add support for **kwargs (user arguments)
        self._set_model_meta_custom_model({"model_meta": model_meta})
        self._load_model(deploy_params)

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
                without_workflow = deploy_params.get("without_workflow", False)
                if without_workflow is False:
                    self._add_workflow_input(model_source, model_files, model_info)
        except Exception as e:
            logger.warning(f"Failed to add input to the workflow: {repr(e)}")

        # remove is_benchmark from deploy_params
        if "without_workflow" in deploy_params:
            deploy_params.pop("without_workflow")

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
            checkpoint_file_info = None
            if not self._is_local_deploy:
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
            if classes is not None:
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
            if (
                self._is_local_deploy
                and os.getenv("SERVER_ADDRESS") is None
                and os.getenv("API_TOKEN") is None
            ):
                return None
            else:
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

    def _create_label(self, dto: PredictionDTO) -> Label:
        raise NotImplementedError("Have to be implemented in child class")

    def _predictions_to_annotation(
        self,
        image_path: Union[str, np.ndarray],
        predictions: List[PredictionDTO],
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

    def api_from_request(self, request) -> Api:
        """
        Get API from request. If not found, use self.api.
        """
        api = request.state.api
        if api is None:
            api = self.api
        return api

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

    def inference(
        self,
        source: Union[str, int, np.ndarray, List[str], List[int], List[np.ndarray]],
        settings: dict = None,
    ) -> Union[Annotation, List[Annotation]]:
        """
        Inference method for images. Provide image path or numpy array of image.

        :param: source: image path,image id, numpy array of image or list of image paths, image ids or numpy arrays
        :type: source: Union[str, int, np.ndarray, List[str], List[int], List[np.ndarray]]
        :param: settings: inference settings
        :type: settings: dict
        :return: annotation or list of annotations
        :rtype: Union[Annotation, List[Annotation]]

        :Usage Example:

            .. code-block:: python

            image_path = "/root/projects/demo/img/sample.jpg"
            ann = model.inference(image_path)
        """
        input_is_list = True
        if not isinstance(source, list):
            input_is_list = False
            source = [source]

        if settings is None:
            settings = self._get_inference_settings({})

        if isinstance(source[0], int):
            results = self.inference_requests_manager.run(
                self._inference_image_ids, self.api, {"batch_ids": source, "settings": settings}
            )
            anns = [
                Annotation.from_json(result["annotation"], self.model_meta) for result in results
            ]
        else:
            anns, _ = self._inference_auto(source, settings)
        if not input_is_list:
            return anns[0]
        return anns

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
    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionDTO]:
        raise NotImplementedError("Have to be implemented in child class")

    def predict_raw(self, image_path: str, settings: Dict[str, Any]) -> List[PredictionDTO]:
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def predict_batch(
        self, images_np: List[np.ndarray], settings: Dict[str, Any]
    ) -> List[List[PredictionDTO]]:
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
    ) -> List[List[PredictionDTO]]:
        """Predict batch of images. `source` is a list of numpy arrays in RGB format"""
        raise NotImplementedError(
            "Have to be implemented in child class If sliding_window_mode is 'advanced'."
        )

    def predict_benchmark(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionDTO]], dict]:
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

    def set_conf_auto(self, conf: float, inference_settings: dict):
        conf_names = ["conf", "confidence", "confidence_threshold", "confidence_thresh"]
        for name in conf_names:
            if name in inference_settings:
                inference_settings[name] = conf
        return inference_settings

    # pylint: enable=method-hidden
    def _get_inference_settings(self, state: dict):
        settings = state.get("settings")
        if settings is None:
            settings = {}
        if "rectangle" in state.keys():
            settings["rectangle"] = state["rectangle"]
        conf = settings.get("conf", None)
        if conf is not None:
            settings = self.set_conf_auto(conf, settings)
        settings["sliding_window_mode"] = self.sliding_window_mode
        for key, value in self.custom_inference_settings_dict.items():
            if key not in settings:
                logger.debug(
                    f"Field {key} not found in inference settings. Use default value {value}"
                )
                settings[key] = value
        return settings

    def _get_batch_size_from_state(self, state: dict):
        batch_size = state.get("batch_size", None)
        if batch_size is None:
            batch_size = self.get_batch_size()
        return batch_size

    @property
    def app(self) -> Application:
        return self._app

    def visualize(
        self,
        predictions: List[PredictionDTO],
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
        predictions: List[Prediction],
    ) -> List[dict]:
        output = [
            {
                **pred.to_json(),
                "data": pred.extra_data.get("slides_data", {}),
            }
            for pred in predictions
        ]
        return output

    def _inference_images(
        self,
        images: Iterable[Union[np.ndarray, str]],
        state: dict,
        inference_request: InferenceRequest,
    ):
        logger.debug("Inferring batch...", extra={"state": state})
        settings = self._get_inference_settings(state)
        logger.debug("Inference settings:", extra={"inference_settings": settings})
        batch_size = self._get_batch_size_from_state(state)

        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, len(images))
        for batch in batched_iter(images, batch_size=batch_size):
            batch = [
                self.cache.get_image_path(image) if isinstance(image, str) else image
                for image in batch
            ]
            anns, slides_data = self._inference_auto(
                batch,
                settings=settings,
            )
            predictions = [Prediction(ann, model_meta=self.model_meta) for ann in anns]
            for pred, this_slides_data in zip(predictions, slides_data):
                pred.extra_data["slides_data"] = this_slides_data
            batch_results = self._format_output(predictions)
            inference_request.add_results(batch_results)
            inference_request.done(len(batch_results))

    def _inference_video(
        self,
        path: str,
        state: Dict,
        inference_request: InferenceRequest,
    ):
        logger.debug("Inferring video...", extra={"path": path, "state": state})
        inference_settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=inference_settings)
        batch_size = self._get_batch_size_from_state(state)
        start_frame_index = state.get("startFrameIndex", 0)
        step = state.get("stride", None)
        if step is None:
            step = state.get("step", None)
        if step is None:
            step = 1
        end_frame_index = state.get("endFrameIndex", None)
        duration = state.get("duration", None)
        frames_count = state.get("framesCount", None)
        tracking = state.get("tracker", None)
        direction = state.get("direction", "forward")
        direction = 1 if direction == "forward" else -1

        frames_reader = VideoFrameReader(path)
        video_height, video_witdth = frames_reader.frame_size()
        if frames_count is not None:
            n_frames = frames_count
        elif end_frame_index is not None:
            n_frames = end_frame_index - start_frame_index + 1
        elif duration is not None:
            fps = frames_reader.fps()
            n_frames = int(duration * fps)
        else:
            n_frames = frames_reader.frames_count()

        if tracking == "bot":
            from supervisely.nn.tracker import BoTTracker

            tracker = BoTTracker(state)
        elif tracking == "deepsort":
            from supervisely.nn.tracker import DeepSortTracker

            tracker = DeepSortTracker(state)
        else:
            if tracking is not None:
                logger.warning(f"Unknown tracking type: {tracking}. Tracking is disabled.")
            tracker = None

        progress_total = (n_frames + step - 1) // step
        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, progress_total)

        results = []
        tracks_data = {}
        for batch in batched(
            range(start_frame_index, start_frame_index + direction * n_frames, direction * step),
            batch_size,
        ):
            if inference_request.is_stopped():
                logger.debug(
                    f"Cancelling inference...",
                    extra={"inference_request_uuid": inference_request.uuid},
                )
                results = []
                break
            logger.debug(
                f"Inferring frames {batch[0]}-{batch[-1]}:",
            )
            frames = frames_reader.read_frames(batch)
            anns, slides_data = self._inference_auto(
                source=frames,
                settings=inference_settings,
            )
            predictions = [
                Prediction(ann, model_meta=self.model_meta, frame_index=frame_index)
                for ann, frame_index in zip(anns, batch)
            ]
            for pred, this_slides_data in zip(predictions, slides_data):
                pred.extra_data["slides_data"] = this_slides_data
            batch_results = self._format_output(predictions)
            if tracker is not None:
                for frame_index, frame, ann in zip(batch, frames, anns):
                    tracks_data = tracker.update(frame, ann, frame_index, tracks_data)
            inference_request.add_results(batch_results)
            inference_request.done(len(batch_results))
            logger.debug(f"Frames {batch[0]}-{batch[-1]} done.")
        video_ann_json = None
        if tracker is not None:
            inference_request.set_stage("Postprocess...", 0, 1)
            video_ann_json = tracker.get_annotation(
                tracks_data, (video_height, video_witdth), n_frames
            ).to_json()
            inference_request.done()
        result = {"ann": results, "video_ann": video_ann_json}
        inference_request.final_result = result.copy()

    def _inference_image_ids(
        self,
        api: Api,
        state: dict,
        inference_request: InferenceRequest,
    ):
        """Inference images by ids.
        If "output_project_id" in state, upload images and annotations to the output project.
        If "output_project_id" equal to source project id, upload annotations to the source project.
        If "output_project_id" is None, write annotations to inference request object.
        """
        logger.debug("Inferring batch_ids", extra={"state": state})
        inference_settings = self._get_inference_settings(state)
        logger.debug("Inference settings:", extra={"inference_settings": inference_settings})
        batch_size = self._get_batch_size_from_state(state)
        image_ids = get_value_for_keys(
            state, ["batch_ids", "image_ids", "images_ids", "imageIds", "image_id", "imageId"]
        )
        if image_ids is None:
            raise ValueError("Image ids are not provided")
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        upload_mode = state.get("upload_mode", None)
        iou_merge_threshold = inference_settings.get("existing_objects_iou_thresh", None)
        if upload_mode == "iou_merge" and iou_merge_threshold is None:
            iou_merge_threshold = 0.7

        images_infos = api.image.get_info_by_id_batch(image_ids)
        images_infos_dict = {im_info.id: im_info for im_info in images_infos}
        inference_request.context.setdefault("image_info", {}).update(images_infos_dict)

        dataset_infos_dict = {
            ds_id: api.dataset.get_info_by_id(ds_id)
            for ds_id in set([im_info.dataset_id for im_info in images_infos])
        }
        inference_request.context.setdefault("dataset_info", {}).update(dataset_infos_dict)

        output_project_id = state.get("output_project_id", None)
        output_dataset_id = None
        inference_request.context.setdefault("project_meta", {})
        if output_project_id is not None:
            if upload_mode is None:
                upload_mode = "append"
        if output_project_id is None and upload_mode == "create":
            image_info = images_infos[0]
            dataset_info = dataset_infos_dict[image_info.dataset_id]
            output_project_info = api.project.create(
                dataset_info.workspace_id,
                name=f"Predictions from task #{self.task_id}",
                description=f"Auto created project from inference request {inference_request.uuid}",
                change_name_if_conflict=True,
            )
            output_project_id = output_project_info.id
            inference_request.context.setdefault("project_info", {})[
                output_project_id
            ] = output_project_info
            output_dataset_info = api.dataset.create(
                output_project_id,
                "Predictions",
                description=f"Auto created dataset from inference request {inference_request.uuid}",
                change_name_if_conflict=True,
            )
            output_dataset_id = output_dataset_info.id
            inference_request.context.setdefault("dataset_info", {})[
                output_dataset_id
            ] = output_dataset_info

        # start download to cache in background
        dataset_image_infos: Dict[int, List[ImageInfo]] = defaultdict(list)
        for image_info in images_infos:
            dataset_image_infos[image_info.dataset_id].append(image_info)
        for dataset_id, ds_image_infos in dataset_image_infos.items():
            self.cache.run_cache_task_manually(
                api, [info.id for info in ds_image_infos], dataset_id=dataset_id
            )

        _upload_predictions = partial(
            self.upload_predictions,
            api=api,
            upload_mode=upload_mode,
            context=inference_request.context,
            dst_dataset_id=output_dataset_id,
            dst_project_id=output_project_id,
            progress_cb=inference_request.done,
            iou_merge_threshold=iou_merge_threshold,
            inference_request=inference_request,
        )

        _add_results_to_request = partial(
            self.add_results_to_request, inference_request=inference_request
        )

        if upload_mode is None:
            upload_f = _add_results_to_request
        else:
            upload_f = _upload_predictions

        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, len(image_ids))
        with Uploader(upload_f, logger=logger) as uploader:
            for image_ids_batch in batched(image_ids, batch_size=batch_size):
                if uploader.has_exception():
                    exception = uploader.exception()
                    raise RuntimeError(f"Error in upload loop: {exception}") from exception
                if inference_request.is_stopped():
                    logger.debug(
                        f"Cancelling inference project...",
                        extra={"inference_request_uuid": inference_request.uuid},
                    )
                    break

                images_nps = [self.cache.download_image(api, img_id) for img_id in image_ids_batch]
                anns, slides_data = self._inference_auto(
                    source=images_nps,
                    settings=inference_settings,
                )

                batch_predictions = []
                for image_id, ann, this_slides_data in zip(image_ids_batch, anns, slides_data):
                    image_info: ImageInfo = images_infos_dict[image_id]
                    dataset_info = dataset_infos_dict[image_info.dataset_id]
                    prediction = Prediction(
                        ann,
                        model_meta=self.model_meta,
                        name=image_info.name,
                        image_id=image_info.id,
                        dataset_id=image_info.dataset_id,
                        project_id=dataset_info.project_id,
                    )
                    prediction.extra_data["slides_data"] = this_slides_data
                    batch_predictions.append(prediction)

                uploader.put(batch_predictions)

    def _inference_video_id(
        self,
        api: Api,
        state: dict,
        inference_request: InferenceRequest,
    ):
        logger.debug("Inferring video_id...", extra={"state": state})
        inference_settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=inference_settings)
        batch_size = self._get_batch_size_from_state(state)
        video_id = get_value_for_keys(state, ["videoId", "video_id"], ignore_none=True)
        if video_id is None:
            raise ValueError("Video id is not provided")
        video_info = api.video.get_info_by_id(video_id)
        start_frame_index = get_value_for_keys(
            state, ["startFrameIndex", "start_frame_index", "start_frame"], ignore_none=True
        )
        if start_frame_index is None:
            start_frame_index = 0
        step = get_value_for_keys(state, ["stride", "step"], ignore_none=True)
        if step is None:
            step = 1
        end_frame_index = get_value_for_keys(
            state, ["endFrameIndex", "end_frame_index", "end_frame"], ignore_none=True
        )
        duration = state.get("duration", None)
        frames_count = get_value_for_keys(
            state, ["framesCount", "frames_count", "num_frames"], ignore_none=True
        )
        tracking = state.get("tracker", None)
        direction = state.get("direction", "forward")
        direction = 1 if direction == "forward" else -1

        if frames_count is not None:
            n_frames = frames_count
        elif end_frame_index is not None:
            n_frames = end_frame_index - start_frame_index
        elif duration is not None:
            fps = video_info.frames_count / video_info.duration
            n_frames = int(duration * fps)
        else:
            n_frames = video_info.frames_count

        if tracking == "bot":
            from supervisely.nn.tracker import BoTTracker

            tracker = BoTTracker(state)
        elif tracking == "deepsort":
            from supervisely.nn.tracker import DeepSortTracker

            tracker = DeepSortTracker(state)
        else:
            if tracking is not None:
                logger.warning(f"Unknown tracking type: {tracking}. Tracking is disabled.")
            tracker = None
        logger.debug(
            f"Video info:",
            extra=dict(
                w=video_info.frame_width,
                h=video_info.frame_height,
                start_frame_index=start_frame_index,
                n_frames=n_frames,
            ),
        )

        # start downloading video in background
        self.cache.run_cache_task_manually(api, None, video_id=video_id)

        progress_total = (n_frames + step - 1) // step
        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, progress_total)

        tracks_data = {}
        for batch in batched(
            range(start_frame_index, start_frame_index + direction * n_frames, direction * step),
            batch_size,
        ):
            if inference_request.is_stopped():
                logger.debug(
                    f"Cancelling inference video...",
                    extra={"inference_request_uuid": inference_request.uuid},
                )
                break
            logger.debug(
                f"Inferring frames {batch[0]}-{batch[-1]}:",
            )
            frames = self.cache.download_frames(api, video_info.id, batch, redownload_video=True)
            anns, slides_data = self._inference_auto(
                source=frames,
                settings=inference_settings,
            )
            predictions = [
                Prediction(
                    ann,
                    model_meta=self.model_meta,
                    frame_index=frame_index,
                    video_id=video_info.id,
                    dataset_id=video_info.dataset_id,
                    project_id=video_info.project_id,
                )
                for ann, frame_index in zip(anns, batch)
            ]
            for pred, this_slides_data in zip(predictions, slides_data):
                pred.extra_data["slides_data"] = this_slides_data
            batch_results = self._format_output(predictions)
            if tracker is not None:
                for frame_index, frame, ann in zip(batch, frames, anns):
                    tracks_data = tracker.update(frame, ann, frame_index, tracks_data)
            inference_request.add_results(batch_results)
            inference_request.done(len(batch_results))
            logger.debug(f"Frames {batch[0]}-{batch[-1]} done.")
        video_ann_json = None
        if tracker is not None:
            inference_request.set_stage("Postprocess...", 0, 1)
            video_ann_json = tracker.get_annotation(
                tracks_data, (video_info.frame_height, video_info.frame_width), n_frames
            ).to_json()
            inference_request.done()
        inference_request.final_result = {"video_ann": video_ann_json}

    def _inference_project_id(self, api: Api, state: dict, inference_request: InferenceRequest):
        """Inference project images.
        If "output_project_id" in state, upload images and annotations to the output project.
        If "output_project_id" equal to source project id, upload annotations to the source project.
        If "output_project_id" is None, write annotations to inference request object.
        """
        logger.debug("Inferring project...", extra={"state": state})
        inference_settings = self._get_inference_settings(state)
        logger.debug("Inference settings:", extra={"inference_settings": inference_settings})
        batch_size = self._get_batch_size_from_state(state)
        project_id = get_value_for_keys(state, keys=["projectId", "project_id"])
        if project_id is None:
            raise ValueError("Project id is not provided")
        project_info = api.project.get_info_by_id(project_id)
        if project_info.type != str(ProjectType.IMAGES):
            raise ValueError("Only images projects are supported.")
        upload_mode = state.get("upload_mode", None)
        iou_merge_threshold = inference_settings.get("existing_objects_iou_thresh", None)
        if upload_mode == "iou_merge" and iou_merge_threshold is None:
            iou_merge_threshold = 0.7
        cache_project_on_model = state.get("cache_project_on_model", False)

        project_info = api.project.get_info_by_id(project_id)
        inference_request.context.setdefault("project_info", {})[project_id] = project_info
        dataset_ids = state.get("dataset_ids", None)
        if dataset_ids is None:
            dataset_ids = state.get("datasetIds", None)
        datasets_infos = api.dataset.get_list(project_info.id, recursive=True)
        inference_request.context.setdefault("dataset_info", {}).update(
            {ds_info.id: ds_info for ds_info in datasets_infos}
        )
        if dataset_ids is not None:
            datasets_infos = [ds_info for ds_info in datasets_infos if ds_info.id in dataset_ids]

        preparing_progress_total = sum([ds_info.items_count for ds_info in datasets_infos])
        inference_progress_total = preparing_progress_total
        inference_request.set_stage(InferenceRequest.Stage.PREPARING, 0, preparing_progress_total)

        output_project_id = state.get("output_project_id", None)
        inference_request.context.setdefault("project_meta", {})
        if output_project_id is not None:
            if upload_mode is None:
                upload_mode = "append"
        if output_project_id is None and upload_mode == "create":
            output_project_info = api.project.create(
                project_info.workspace_id,
                name=f"Predictions from task #{self.task_id}",
                description=f"Auto created project from inference request {inference_request.uuid}",
                change_name_if_conflict=True,
            )
            output_project_id = output_project_info.id
            inference_request.context.setdefault("project_info", {})[
                output_project_id
            ] = output_project_info

        if cache_project_on_model:
            download_to_cache(
                api, project_info.id, datasets_infos, progress_cb=inference_request.done
            )

        images_infos_dict = {}
        for dataset_info in datasets_infos:
            images_infos_dict[dataset_info.id] = api.image.get_list(dataset_info.id)
            if not cache_project_on_model:
                inference_request.done(dataset_info.items_count)

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

        _upload_predictions = partial(
            self.upload_predictions,
            api=api,
            upload_mode=upload_mode,
            context=inference_request.context,
            dst_project_id=output_project_id,
            progress_cb=inference_request.done,
            iou_merge_threshold=iou_merge_threshold,
            inference_request=inference_request,
        )

        _add_results_to_request = partial(
            self.add_results_to_request, inference_request=inference_request
        )

        if upload_mode is None:
            upload_f = _add_results_to_request
        else:
            upload_f = _upload_predictions

        inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, inference_progress_total)
        with Uploader(upload_f, logger=logger) as uploader:
            for dataset_info in datasets_infos:
                for images_infos_batch in batched(
                    images_infos_dict[dataset_info.id], batch_size=batch_size
                ):
                    if inference_request.is_stopped():
                        logger.debug(
                            f"Cancelling inference project...",
                            extra={"inference_request_uuid": inference_request.uuid},
                        )
                        return
                    if uploader.has_exception():
                        exception = uploader.exception
                        raise RuntimeError(f"Error in upload loop: {exception}") from exception
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
                        settings=inference_settings,
                    )
                    predictions = [
                        Prediction(
                            ann,
                            model_meta=self.model_meta,
                            image_id=image_info.id,
                            name=image_info.name,
                            dataset_id=dataset_info.id,
                            project_id=dataset_info.project_id,
                            image_name=image_info.name,
                        )
                        for ann, image_info in zip(anns, images_infos_batch)
                    ]
                    for pred, this_slides_data in zip(predictions, slides_data):
                        pred.extra_data["slides_data"] = this_slides_data

                    uploader.put(predictions)

    def _run_speedtest(
        self,
        api: Api,
        state: dict,
        inference_request: InferenceRequest,
    ):
        """Run speedtest on project images."""
        logger.debug("Running speedtest...", extra={"state": state})
        settings = self._get_inference_settings(state)
        logger.debug(f"Inference settings:", extra=settings)

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

        preparing_progress_total = len(datasets_infos)
        if cache_project_on_model:
            preparing_progress_total += sum(
                dataset_info.items_count for dataset_info in datasets_infos
            )
        inference_request.set_stage(InferenceRequest.Stage.PREPARING, 0, preparing_progress_total)

        images_infos_dict = {}
        for dataset_info in datasets_infos:
            images_infos_dict[dataset_info.id] = api.image.get_list(dataset_info.id)
            inference_request.done()

        if cache_project_on_model:
            download_to_cache(api, project_id, datasets_infos, progress_cb=inference_request.done)

        inference_request.set_stage("warmup", 0, num_warmup)

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

        def upload_f(benchmarks: List):
            inference_request.add_results(benchmarks)
            inference_request.done(len(benchmarks))

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

        with Uploader(upload_f=upload_f, logger=logger) as uploader:
            for i in range(num_iterations + num_warmup):
                if inference_request.is_stopped():
                    logger.debug(
                        f"Cancelling inference project...",
                        extra={"inference_request_uuid": inference_request.uuid},
                    )
                    return
                if uploader.has_exception():
                    exception = uploader.exception
                    raise RuntimeError(f"Error in upload loop: {exception}") from exception
                if i == num_warmup:
                    inference_request.set_stage(InferenceRequest.Stage.INFERENCE, 0, num_iterations)

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
                    uploader.put([benchmark])
                else:
                    inference_request.done()

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

    def _deploy_on_autorestart(self):
        try:
            self._api_request_model_layout._title = (
                "Model was deployed during auto restart with the following settings"
            )
            self._api_request_model_layout.update_data()
            deploy_params = self.autorestart.deploy_params
            if isinstance(self.gui, GUI.ServingGUITemplate):
                model_files = self._download_model_files(deploy_params)
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

    def validate_inference_state(self, state: Union[Dict, str], log_error=True):
        try:
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except (json.decoder.JSONDecodeError, TypeError) as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot decode settings: {e}",
                    )
            if not isinstance(state, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Settings is not json object"
                )
            batch_size = state.get("batch_size", None)
            if batch_size is None:
                batch_size = self.get_batch_size()
            if self.max_batch_size is not None and batch_size > self.max_batch_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Batch size should be less than or equal to {self.max_batch_size} for this model.",
                )
        except Exception as e:
            if log_error:
                logger.error(f"Error validating request state: {e}", exc_info=True)
            raise

    def upload_predictions(
        self,
        predictions: List[Prediction],
        api: Api,
        upload_mode: str,
        context: Dict = None,
        dst_dataset_id: int = None,
        dst_project_id: int = None,
        progress_cb=None,
        iou_merge_threshold: float = None,
        inference_request: InferenceRequest = None,
    ):
        ds_predictions: Dict[int, List[Prediction]] = defaultdict(list)
        for prediction in predictions:
            ds_predictions[prediction.dataset_id].append(prediction)

        def _new_name(image_info: ImageInfo):
            name = Path(image_info.name)
            stem = name.stem
            parent = name.parent
            suffix = name.suffix
            return str(parent / f"{stem}(dataset_id:{image_info.dataset_id}){suffix}")

        def _get_or_create_dataset(src_dataset_id, dst_project_id):
            if src_dataset_id is None:
                return None
            created_dataset_id = context.setdefault("created_dataset", {}).get(src_dataset_id, None)
            if created_dataset_id is not None:
                return created_dataset_id
            src_dataset_info: DatasetInfo = context.setdefault("dataset_info", {}).get(
                src_dataset_id
            )
            if src_dataset_info is None:
                src_dataset_info = api.dataset.get_info_by_id(src_dataset_id)
                context["dataset_info"][src_dataset_id] = src_dataset_info
            src_parent_id = src_dataset_info.parent_id
            dst_parent_id = _get_or_create_dataset(src_parent_id, dst_project_id)
            created_dataset = api.dataset.create(
                dst_project_id,
                src_dataset_info.name,
                description=f"Auto created dataset from inference request {inference_request.uuid if inference_request is not None else ''}",
                change_name_if_conflict=True,
                parent_id=dst_parent_id,
            )
            context["dataset_info"][created_dataset.id] = created_dataset
            context.setdefault("created_dataset", {})[src_dataset_id] = created_dataset.id
            return created_dataset.id

        created_names = []
        if context is None:
            context = {}
        for dataset_id, preds in ds_predictions.items():
            if dst_project_id is not None:
                # upload to the destination project
                dst_dataset_id = _get_or_create_dataset(
                    src_dataset_id=dataset_id, dst_project_id=dst_project_id
                )
            if dst_dataset_id is not None:
                # upload to the destination dataset
                dataset_info = context.setdefault("dataset_info", {}).get(dst_dataset_id, None)
                if dataset_info is None:
                    dataset_info = api.dataset.get_info_by_id(dst_dataset_id)
                    context["dataset_info"][dst_dataset_id] = dataset_info
                project_id = dataset_info.project_id
                project_meta = context.setdefault("project_meta", {}).get(project_id, None)
                if project_meta is None:
                    project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
                    context["project_meta"][project_id] = project_meta

                meta_changed = False
                for pred in preds:
                    ann = pred.annotation
                    project_meta, ann, meta_changed_ = update_meta_and_ann(project_meta, ann)
                    meta_changed = meta_changed or meta_changed_
                    pred.annotation = ann
                    prediction.model_meta = project_meta

                if meta_changed:
                    project_meta = api.project.update_meta(project_id, project_meta)
                    context["project_meta"][project_id] = project_meta

                anns = _exclude_duplicated_predictions(
                    api,
                    [pred.annotation for pred in preds],
                    dataset_id,
                    [pred.image_id for pred in preds],
                    iou=iou_merge_threshold,
                    meta=project_meta,
                )
                for pred, ann in zip(preds, anns):
                    pred.annotation = ann

                context.setdefault("image_info", {})
                missing = [
                    pred.image_id for pred in preds if pred.image_id not in context["image_info"]
                ]
                if missing:
                    context["image_info"].update(
                        {
                            image_info.id: image_info
                            for image_info in api.image.get_info_by_id_batch(missing)
                        }
                    )
                image_infos: List[ImageInfo] = [
                    context["image_info"][pred.image_id] for pred in preds
                ]
                dst_names = [
                    _new_name(image_info) if image_info.name in created_names else image_info.name
                    for image_info in image_infos
                ]
                dst_image_infos = api.image.copy_batch_optimized(
                    dataset_id,
                    image_infos,
                    dst_dataset_id,
                    dst_names=dst_names,
                    with_annotations=False,
                    save_source_date=False,
                )
                created_names.extend([image_info.name for image_info in dst_image_infos])
                api.annotation.upload_anns([image_info.id for image_info in dst_image_infos], anns)
            else:
                # upload to the source dataset
                ds_info = context.setdefault("dataset_info", {}).get(dataset_id, None)
                if ds_info is None:
                    ds_info = api.dataset.get_info_by_id(dataset_id)
                    context["dataset_info"][dataset_id] = ds_info
                project_id = ds_info.project_id

                project_meta = context.setdefault("project_meta", {}).get(project_id, None)
                if project_meta is None:
                    project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
                    context["project_meta"][project_id] = project_meta

                meta_changed = False
                for pred in preds:
                    ann = pred.annotation
                    project_meta, ann, meta_changed_ = update_meta_and_ann(project_meta, ann)
                    meta_changed = meta_changed or meta_changed_
                    pred.annotation = ann
                    prediction.model_meta = project_meta

                if meta_changed:
                    project_meta = api.project.update_meta(project_id, project_meta)
                    context["project_meta"][project_id] = project_meta

                anns = _exclude_duplicated_predictions(
                    api,
                    [pred.annotation for pred in preds],
                    dataset_id,
                    [pred.image_id for pred in preds],
                    iou=iou_merge_threshold,
                    meta=project_meta,
                )
                for pred, ann in zip(preds, anns):
                    pred.annotation = ann

                if upload_mode in ["iou_merge", "append"]:
                    context.setdefault("annotation", {})
                    missing = []
                    for pred in preds:
                        if pred.image_id not in context["annotation"]:
                            missing.append(pred.image_id)
                    for image_id, ann_info in zip(
                        missing, api.annotation.download_batch(dataset_id, missing)
                    ):
                        context["annotation"][image_id] = Annotation.from_json(
                            ann_info.annotation, project_meta
                        )
                    for pred in preds:
                        pred.annotation = context["annotation"][pred.image_id].merge(
                            pred.annotation
                        )

                api.annotation.upload_anns(
                    [pred.image_id for pred in preds],
                    [pred.annotation for pred in preds],
                )

            if progress_cb is not None:
                progress_cb(len(preds))

        if inference_request is not None:
            results = self._format_output(predictions)
            for result in results:
                result["annotation"] = None
                result["data"] = None
            inference_request.add_results(results)

    def add_results_to_request(
        self, predictions: List[Prediction], inference_request: InferenceRequest
    ):
        results = self._format_output(predictions)
        inference_request.add_results(results)
        inference_request.done(len(results))

    def serve(self):
        if not self._use_gui and not self._is_local_deploy:
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

        if self._is_local_deploy:
            # Predict and shutdown
            if self._args.mode == "predict" and any(
                [
                    self._args.input,
                    self._args.project_id,
                    self._args.dataset_id,
                    self._args.image_id,
                ]
            ):

                self._parse_inference_settings_from_args()
                self._inference_by_local_deploy_args()
                exit(0)

        if isinstance(self.gui, GUI.InferenceGUI):
            self._app = Application(layout=self.get_ui())
        elif isinstance(self.gui, GUI.ServingGUI):
            self._app = Application(layout=self._app_layout)
        # elif isinstance(self.gui, GUI.InferenceGUI):
        #     self._app = Application(layout=self.get_ui())
        else:
            self._app = Application(layout=self.get_ui())

        if self._task_id is not None and is_production():
            try:
                response = self.api.task.get_fields(
                    self._task_id, [AutoRestartInfo.Fields.AUTO_RESTART_INFO]
                )
                self.autorestart = AutoRestartInfo.from_response(response)
                if self.autorestart is not None:
                    logger.debug("Autorestart info is set.", extra=self.autorestart.deploy_params)
                    self._deploy_on_autorestart()
                else:
                    logger.debug("Autorestart info is not set.")
            except Exception:
                logger.error("Autorestart failed.", exc_info=True)

        server = self._app.get_server()
        self._app.set_ready_check_function(self.is_model_deployed)

        if self.api is not None:

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

        @server.exception_handler(HTTPException)
        def http_exception_handler(request: Request, exc: HTTPException):
            response_content = {
                "detail": exc.detail,
                "success": False,
            }
            if isinstance(exc.detail, dict):
                if "message" in exc.detail:
                    response_content["message"] = exc.detail["message"]
                if "success" in exc.detail:
                    response_content["success"] = exc.detail["success"]
            elif isinstance(exc.detail, str):
                response_content["message"] = exc.detail

            return JSONResponse(status_code=exc.status_code, content=response_content)

        self.cache.add_cache_endpoint(server)
        self.cache.add_cache_files_endpoint(server)

        @server.post(f"/get_session_info")
        @self._check_serve_before_call
        def get_session_info(response: Response):
            return self.get_info()

        @server.post("/get_custom_inference_settings")
        def get_custom_inference_settings():
            return {"settings": self.custom_inference_settings}

        @server.post("/get_model_meta")
        @server.post("/get_output_classes_and_tags")
        def get_output_classes_and_tags():
            return self.model_meta.to_json()

        @server.post("/inference_image_id")
        def inference_image_id(request: Request):
            state = request.state.state
            logger.debug("Received a request to '/inference_image_id'", extra={"state": state})
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            return self.inference_requests_manager.run(self._inference_image_ids, api, state)[0]

        @server.post("/inference_image_id_async")
        def inference_image_id_async(request: Request):
            state = request.state.state
            logger.debug(
                "Received a request to 'inference_image_id_async'",
                extra={"state": state},
            )
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._inference_image_ids,
                api,
                state,
            )
            return {
                "message": "Scheduled inference task.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post("/inference_image")
        def inference_image(
            files: List[UploadFile], settings: str = Form("{}"), state: str = Form("{}")
        ):
            if state == "{}" or not state:
                state = settings
            state = str(state)
            logger.debug("Received a request to 'inference_image'", extra={"state": state})
            self.validate_inference_state(state)
            state = json.loads(state)
            if len(files) != 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Only one file expected but got {len(files)}",
                )
            try:
                file = files[0]
                inference_request = self.inference_requests_manager.create()
                inference_request.set_stage(InferenceRequest.Stage.PREPARING, 0, file.size)

                img_bytes = b""
                while buf := file.read(64 * 1024 * 1024):
                    img_bytes += buf
                    inference_request.done(len(buf))

                image = sly_image.read_bytes(img_bytes)
                inference_request, future = self.inference_requests_manager.schedule_task(
                    self._inference_images, [image], state, inference_request=inference_request
                )
                future.result()
                return inference_request.pop_pending_results()[0]
            except sly_image.UnsupportedImageFormat:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File has unsupported format. Supported formats: {sly_image.SUPPORTED_IMG_EXTS}",
                )

        @server.post("/inference_image_url")
        def inference_image_url(request: Request):
            state = request.state.state
            logger.debug("Received a request to 'inference_image_url'", extra={"state": state})
            self.validate_inference_state(state)
            image_url = state["image_url"]
            ext = sly_fs.get_file_ext(image_url)
            if ext == "":
                ext = ".jpg"
            with requests.get(image_url, stream=True) as response:
                response.raise_for_status()
                response.raw.decode_content = True
                image = self.cache.add_image_to_cache(image_url, response.raw, ext=ext)
            return self.inference_requests_manager.run(self._inference_images, [image], state)[0]

        @server.post("/inference_batch_ids")
        def inference_batch_ids(request: Request):
            state = request.state.state
            logger.debug("Received a request to  'inference_batch_ids'", extra={"state": state})
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            return self.inference_requests_manager.run(self._inference_image_ids, api, state)

        @server.post("/inference_batch_ids_async")
        def inference_batch_ids_async(request: Request):
            state = request.state.state
            logger.debug(
                f"Received a request to 'inference_batch_ids_async'", extra={"state": state}
            )
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._inference_image_ids, api, state
            )
            return {
                "message": "Scheduled inference task.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post("/inference_batch")
        def inference_batch(
            response: Response,
            files: List[UploadFile],
            settings: str = Form("{}"),
            state: str = Form("{}"),
        ):
            if state == "{}" or not state:
                state = settings
            state = str(state)
            logger.debug("Received a request to 'inference_batch'", extra={"state": state})
            self.validate_inference_state(state)
            state = json.loads(state)
            if len(files) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"At least one file is expected but got {len(files)}",
                )
            try:
                inference_request = self.inference_requests_manager.create()
                inference_request.set_stage(
                    InferenceRequest.Stage.PREPARING, 0, sum([file.size for file in files])
                )

                names = []
                for file in files:
                    name = file.filename
                    if name is None or name == "":
                        name = rand_str(10)
                    ext = Path(name).suffix
                    img_bytes = b""
                    while buf := file.file.read(64 * 1024 * 1024):
                        img_bytes += buf
                        inference_request.done(len(buf))
                    self.cache.add_image_to_cache(name, img_bytes, ext=ext)
                    names.append(name)

                inference_request, future = self.inference_requests_manager.schedule_task(
                    self._inference_images, names, state, inference_request=inference_request
                )
                future.result()
                return inference_request.pop_pending_results()
            except sly_image.UnsupportedImageFormat:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"File has unsupported format. Supported formats: {sly_image.SUPPORTED_IMG_EXTS}"

        @server.post("/inference_batch_async")
        def inference_batch_async(
            response: Response,
            files: List[UploadFile],
            settings: str = Form("{}"),
            state: str = Form("{}"),
        ):
            if state == "{}" or not state:
                state = settings
            state = str(state)
            logger.debug("Received a request to 'inference_batch'", extra={"state": state})
            self.validate_inference_state(state)
            state = json.loads(state)
            if len(files) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"At least one file is expected but got {len(files)}",
                )
            try:
                inference_request = self.inference_requests_manager.create()
                inference_request.set_stage(
                    InferenceRequest.Stage.PREPARING, 0, sum([file.size for file in files])
                )

                names = []
                for file in files:
                    name = file.filename
                    if name is None or name == "":
                        name = rand_str(10)
                    ext = Path(name).suffix
                    img_bytes = b""
                    while buf := file.file.read(64 * 1024 * 1024):
                        img_bytes += buf
                        inference_request.done(len(buf))
                    self.cache.add_image_to_cache(name, img_bytes, ext=ext)
                    names.append(name)

                inference_request, _ = self.inference_requests_manager.schedule_task(
                    self._inference_images, names, state, inference_request=inference_request
                )
                return {
                    "message": "Scheduled inference task.",
                    "inference_request_uuid": inference_request.uuid,
                }
            except sly_image.UnsupportedImageFormat:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return f"File has unsupported format. Supported formats: {sly_image.SUPPORTED_IMG_EXTS}"

        @server.post("/inference_video_id")
        def inference_video_id(request: Request):
            state = request.state.state
            logger.debug(f"Received a request to 'inference_video_id'", extra={"state": state})
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            inference_request, future = self.inference_requests_manager.schedule_task(
                self._inference_video_id, api, state
            )
            future.result()
            results = {"ann": inference_request.pop_pending_results()}
            final_result = inference_request.final_result
            if final_result is not None:
                results.update(final_result)
            return results

        @server.post("/inference_video_async")
        def inference_video_async(
            files: List[UploadFile],
            settings: str = Form("{}"),
            state: str = Form("{}"),
        ):
            if state == "{}" or not state:
                state = settings
            state = str(state)
            logger.debug("Received a request to 'inference_video_async'", extra={"state": state})
            self.validate_inference_state(state)
            state = json.loads(state)

            file = files[0]
            video_name = files[0].filename
            video_source = files[0].file
            file_size = file.size

            inference_request = self.inference_requests_manager.create()
            inference_request.set_stage(InferenceRequest.Stage.PREPARING, 0, file_size)

            video_source.read = progress_wrapper(
                video_source.read, inference_request.progress.iters_done_report
            )

            if self.cache.is_persistent:
                self.cache.add_video_to_cache(video_name, video_source)
                video_path = self.cache.get_video_path(video_name)
            else:
                video_path = os.path.join(tempfile.gettempdir(), video_name)
                with open(video_path, "wb") as video_file:
                    shutil.copyfileobj(
                        video_source, open(video_path, "wb"), length=(64 * 1024 * 1024)
                    )

            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._inference_video,
                path=video_path,
                state=state,
                inference_request=inference_request,
            )

            return {
                "message": "Scheduled inference task.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post("/inference_video_id_async")
        def inference_video_id_async(response: Response, request: Request):
            state = request.state.state
            logger.debug("Received a request to 'inference_video_id_async'", extra={"state": state})
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._inference_video_id, api, state
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post("/inference_project_id_async")
        def inference_project_id_async(response: Response, request: Request):
            state = request.state.state
            logger.debug(
                "Received a request to 'inference_project_id_async'", extra={"state": state}
            )
            self.validate_inference_state(state)
            api = self.api_from_request(request)
            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._inference_project_id, api, state
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post("/run_speedtest")
        def run_speedtest(response: Response, request: Request):
            state = request.state.state
            logger.debug(f"'run_speedtest' request in json format:{state}")

            batch_size = state["batch_size"]
            if batch_size > 1 and not self.is_batch_inference_supported():
                response.status_code = status.HTTP_501_NOT_IMPLEMENTED
                return {
                    "message": "Batch inference is not implemented for this model.",
                    "success": False,
                }

            self.validate_inference_state(state)
            api = self.api_from_request(request)

            project_id = state["projectId"]
            project_info = api.project.get_info_by_id(project_id)
            if project_info.type != str(ProjectType.IMAGES):
                response.status_code = status.HTTP_400_BAD_REQUEST
                response.body = {"message": "Only images projects are supported."}
                raise ValueError("Only images projects are supported.")

            inference_request, _ = self.inference_requests_manager.schedule_task(
                self._run_speedtest, api, state
            )
            return {
                "message": "Inference has started.",
                "inference_request_uuid": inference_request.uuid,
            }

        @server.post(f"/get_inference_progress")
        def get_inference_progress(response: Response, request: Request):
            state = request.state.state
            logger.debug("Received a request to '/get_inference_progress'", extra={"state": state})
            inference_request_uuid = state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            inference_request = self.inference_requests_manager.get(inference_request_uuid)
            log_extra = _get_log_extra_for_inference_request(
                inference_request.uuid, inference_request
            )
            data = {**inference_request.to_json(), **log_extra}
            if inference_request.stage != InferenceRequest.Stage.INFERENCE:
                data["progress"] = {"current": 0, "total": 1}
            logger.debug(
                f"Sending inference progress with uuid:",
                extra=data,
            )
            return data

        @server.post(f"/pop_inference_results")
        def pop_inference_results(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            if inference_request_uuid in self._inference_requests:
                inference_request = self._inference_requests[inference_request_uuid].copy()
                inference_request["pending_results"] = inference_request["pending_results"].copy()

                # Clear the queue `pending_results`
                self._inference_requests[inference_request_uuid]["pending_results"].clear()

                inference_request["progress"] = _convert_sly_progress_to_dict(
                    inference_request["progress"]
                )
                log_extra = _get_log_extra_for_inference_request(
                    inference_request_uuid, inference_request
                )
                logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
                return inference_request

            inference_request = self.inference_requests_manager.get(inference_request_uuid)
            log_extra = _get_log_extra_for_inference_request(
                inference_request.uuid, inference_request
            )
            data = {
                **inference_request.to_json(),
                **log_extra,
                "pending_results": inference_request.pop_pending_results(),
            }

            logger.debug(f"Sending inference delta results with uuid:", extra=log_extra)
            return data

        @server.post(f"/get_inference_result")
        def get_inference_result(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            if inference_request_uuid in self._inference_requests:
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

            inference_request = self.inference_requests_manager.get(inference_request_uuid)
            log_extra = _get_log_extra_for_inference_request(
                inference_request.uuid, inference_request
            )
            logger.debug(
                f"Sending inference result with uuid:",
                extra=log_extra,
            )

            return inference_request.final_result

        @server.post(f"/stop_inference")
        def stop_inference(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {
                    "message": "Error: 'inference_request_uuid' is required.",
                    "success": False,
                }
            if inference_request_uuid in self._inference_requests:
                inference_request = self._inference_requests[inference_request_uuid]
                inference_request["cancel_inference"] = True
            else:
                inference_request = self.inference_requests_manager.get(inference_request_uuid)
                inference_request.stop()
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
            if inference_request_uuid in self._inference_requests:
                del self._inference_requests[inference_request_uuid]
            else:
                self.inference_requests_manager.remove_after(inference_request_uuid, 60)
            logger.debug("Removed an inference request:", extra={"uuid": inference_request_uuid})
            return {"success": True}

        @server.post(f"/get_preparing_progress")
        def get_preparing_progress(response: Response, request: Request):
            inference_request_uuid = request.state.state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}

            if inference_request_uuid in self._inference_requests:
                inference_request = self._inference_requests[inference_request_uuid].copy()
                return inference_request["preparing_progress"]
            inference_request = self.inference_requests_manager.get(inference_request_uuid)
            return _get_log_extra_for_inference_request(inference_request.uuid, inference_request)[
                "preparing_progress"
            ]

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
                model_name = state.get("model_name", None)
                if isinstance(self.gui, GUI.ServingGUITemplate):
                    if deploy_params["model_source"] == ModelSource.PRETRAINED and model_name:
                        deploy_params = self._build_deploy_params_from_api(
                            model_name, deploy_params
                        )
                    model_files = self._download_model_files(deploy_params)
                    deploy_params["model_files"] = model_files
                    deploy_params = self._set_common_deploy_params(deploy_params)
                    self._load_model_headless(**deploy_params)
                elif isinstance(self.gui, GUI.ServingGUI):
                    if deploy_params["model_source"] == ModelSource.PRETRAINED and model_name:
                        deploy_params = self._build_legacy_deploy_params_from_api(model_name)
                    deploy_params = self._set_common_deploy_params(deploy_params)
                    self._load_model(deploy_params)
                elif self.gui is None and self.api is None:
                    if deploy_params["model_source"] == ModelSource.PRETRAINED and model_name:
                        deploy_params = self._build_deploy_params_from_api(
                            model_name, deploy_params
                        )
                        model_files = self._download_model_files(deploy_params)
                        deploy_params["model_files"] = model_files

                    deploy_params = self._set_common_deploy_params(deploy_params)
                    self._load_model_headless(**deploy_params)
                    logger.info(
                        f"Model has been successfully loaded on {deploy_params['device']} device"
                    )
                    return {"result": "model was successfully deployed"}

                else:
                    raise ValueError("Unknown GUI type")
                if self.gui is not None:
                    self.set_params_to_gui(deploy_params)
                    # update to set correct device
                    device = deploy_params.get("device", "cpu")
                    self.gui.set_deployed(device)
                return {"result": "model was successfully deployed"}
            except Exception as e:
                if self.gui is not None:
                    self.gui._success_label.hide()
                raise e

        @server.post("/list_pretrained_models")
        def _list_pretrained_models():
            if isinstance(self.gui, GUI.ServingGUITemplate):
                return [
                    _get_model_name(model) for model in self._gui.pretrained_models_table._models
                ]
            elif hasattr(self, "pretrained_models"):
                return [_get_model_name(model) for model in self.pretrained_models]
            else:
                if hasattr(self, "pretrained_models_table"):
                    return [
                        _get_model_name(model)
                        for model in self.pretrained_models_table._models  # pylint: disable=no-member
                    ]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Pretrained models table is not available in this app.",
                    )

        @server.post("/list_pretrained_model_infos")
        def _list_pretrained_model_infos():
            if isinstance(self.gui, GUI.ServingGUITemplate):
                return self._gui.pretrained_models_table._models
            elif hasattr(self, "pretrained_models"):
                return self.pretrained_models
            else:
                if hasattr(self, "pretrained_models_table"):
                    return self.pretrained_models_table._models
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Pretrained models table is not available in this app.",
                    )

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

        @server.post("/get_inference_status")
        def _get_inference_status(request: Request, response: Response):
            state = request.state.state
            inference_request_uuid = state.get("inference_request_uuid")
            if inference_request_uuid is None:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "Error: 'inference_request_uuid' is required."}
            inference_request = self.inference_requests_manager.get(inference_request_uuid)
            if inference_request is None:
                response.status_code = status.HTTP_404_NOT_FOUND
                return {"message": "Error: 'inference_request_uuid' is not found."}
            return inference_request.status()

        @server.post("/get_status")
        def _get_status(request: Request):
            progress = self.inference_requests_manager.global_progress.to_json()
            ram_allocated, ram_total = get_ram_usage()
            gpu_allocated, gpu_total = get_gpu_usage()
            return {
                "is_deployed": self.is_model_deployed(),
                "progress": progress,
                "gpu_memory": {
                    "allocated": gpu_allocated,
                    "total": gpu_total,
                },
                "ram_memory": {
                    "allocated": ram_allocated,
                    "total": ram_total,
                },
            }

        # Local deploy without predict args
        if self._is_local_deploy:
            self._run_server()

    def _parse_local_deploy_args(self):
        parser = argparse.ArgumentParser(description="Run Inference Serving")

        # Positional args
        parser.add_argument(
            "mode", nargs="?", type=str, help="Mode of operation: 'deploy' or 'predict'"
        )
        parser.add_argument("input", nargs="?", type=str, help="Local path to input data")

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
            choices=[
                RuntimeType.PYTORCH,
                RuntimeType.ONNXRUNTIME,
                RuntimeType.TENSORRT,
            ],
            default=RuntimeType.PYTORCH,
            help="Runtime type for inference (default: PYTORCH)",
        )
        # -------------------------- #

        # Remote predict
        parser.add_argument(
            "--project_id",
            type=int,
            required=False,
            help="Project ID on Supervisely instance",
        )
        parser.add_argument(
            "--dataset_id",
            type=lambda x: [int(i) for i in x.split(",")] if "," in x else int(x),
            required=False,
            help="ID of the dataset or a comma-separated list of dataset IDs e.g. '505,506,507'",
        )
        parser.add_argument(
            "--image_id",
            type=int,
            required=False,
            help="Image ID on Supervisely instance",
        )
        # -------------------------- #

        # Output args
        parser.add_argument(
            "--output",
            type=str,
            required=False,
            help="Path to local directory where predictions will be saved. Default: './predictions'",
        )
        parser.add_argument(
            "--upload",
            required=False,
            action="store_true",
            help="Upload predictions to Supervisely instance. Works only with: '--project_id', '--dataset_id', '--image_id'. For project and dataset predictions a new project will be created. Default: False",
        )
        # -------------------------- #

        # Other args
        parser.add_argument(
            "--settings",
            type=str,
            required=False,
            nargs="*",
            help="Path to the settings JSON/YAML file or key=value pairs",
        )
        parser.add_argument(
            "--draw",
            required=False,
            action="store_true",
            help="Generate new images with visualized predictions. Default: False",
        )
        # -------------------------- #

        # Parse arguments
        args, _ = parser.parse_known_args()
        if args.mode is None:
            return None, False
        elif args.mode not in ["predict", "deploy"]:
            return None, False

        if args.model is None:
            if len(self.pretrained_models) == 0:
                raise ValueError("No pretrained models found.")

            model = self.pretrained_models[0]
            model_name = _get_model_name(model)
            if model_name is None:
                raise ValueError("No model name found in the first pretrained model.")

            args.model = model_name
            logger.info(
                f"Argument '--model' is not provided. Model: '{model_name}' will be deployed."
            )
        if args.mode not in ["deploy", "predict"]:
            raise ValueError("Invalid operation. Only 'deploy' or 'predict' is supported.")
        if args.output is None:
            args.output = "./predictions"
        if isinstance(args.dataset_id, int):
            args.dataset_id = [args.dataset_id]

        return args, True

    def _parse_inference_settings_from_args(self):
        def parse_value(value: str):
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            if value.lower() == ("none", "null"):
                return None
            if value.isdigit():
                return int(value)
            if "." in value:
                parts = value.split(".")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return float(value)
            return value

        args = self._args
        # Parse settings argument
        settings_dict = {}
        if args.settings:
            is_settings_file = args.settings[0].endswith((".json", ".yaml", ".yml"))
            if len(args.settings) == 1 and is_settings_file:
                args.settings = args.settings[0]
            else:
                for setting in args.settings:
                    if "=" in setting:
                        key, value = setting.split("=", 1)
                        settings_dict[key] = parse_value(value)
                    elif ":" in setting:
                        key, value = setting.split(":", 1)
                        settings_dict[key] = parse_value(value)
                    else:
                        raise ValueError(
                            f"Invalid setting: '{setting}'. Please use key value pairs separated by '=', e.g. conf=0.4'"
                        )
                args.settings = settings_dict
        args.settings = self._read_settings(args.settings)
        self._validate_settings(args.settings)

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
            model_name = _get_model_name(m)
            if model_name is None:
                continue
            m_files = meta.get("model_files", None)
            if m_files is None:
                continue
            checkpoint = m_files.get("checkpoint", None)
            if checkpoint is None:
                continue
            if model.lower() == model_name.lower():
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
            model_meta_path = os.path.join(artifacts_dir, "model_meta.json")
            model_info["model_meta"] = self._load_json_file(model_meta_path)
            original_model_files = model_info.get("model_files")
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

        logger.debug(f"Deploy parameters: {deploy_params}")
        return deploy_params, need_download

    def _run_server(self):
        config = uvicorn.Config(app=self._app, host="0.0.0.0", port=8000, ws="websockets")
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()

    def _read_settings(self, settings: Union[str, Dict[str, Any]]):
        if isinstance(settings, dict):
            return settings

        settings_path = settings
        if settings_path is None:
            return {}
        if settings_path.endswith(".json"):
            return sly_json.load_json_file(settings_path)
        elif settings_path.endswith(".yaml") or settings_path.endswith(".yml"):
            with open(settings_path, "r") as f:
                return yaml.safe_load(f)
        raise ValueError("Settings file should be in JSON or YAML format")

    def _validate_settings(self, settings: dict):
        default_settings = self.custom_inference_settings_dict
        if settings == {}:
            self._args.settings = default_settings
            return
        for key, value in settings.items():
            if key not in default_settings and key != "classes":
                acceptable_keys = ", ".join(default_settings.keys()) + ", 'classes'"
                raise ValueError(
                    f"Inference settings doesn't have key: '{key}'. Available keys are: '{acceptable_keys}'"
                )

    def _inference_by_local_deploy_args(self):
        missing_env_message = "Set 'SERVER_ADDRESS' and 'API_TOKEN' environment variables to predict data on Supervisely platform."

        def predict_project_id_by_args(
            api: Api,
            project_id: int,
            dataset_ids: List[int] = None,
            output_dir: str = "./predictions",
            settings: str = None,
            draw: bool = False,
            upload: bool = False,
        ):
            if self.api is None:
                raise ValueError(missing_env_message)

            if dataset_ids:
                logger.info(f"Predicting datasets: '{dataset_ids}'")
            else:
                logger.info(f"Predicting project: '{project_id}'")

            if draw:
                raise ValueError("Draw visualization is not supported for project inference")

            state = {
                "projectId": project_id,
                "dataset_ids": dataset_ids,
                "settings": settings,
            }
            if upload:
                source_project = api.project.get_info_by_id(project_id)
                workspace_id = source_project.workspace_id
                output_project = api.project.create(
                    workspace_id,
                    f"{source_project.name} predicted",
                    change_name_if_conflict=True,
                )
                state["output_project_id"] = output_project.id
            results = self.inference_requests_manager.run(self._inference_project_id, api, state)

            dataset_infos = api.dataset.get_list(project_id)
            datasets_map = {dataset_info.id: dataset_info.name for dataset_info in dataset_infos}

            if not upload:
                for prediction in results:
                    dataset_name = datasets_map[prediction["dataset_id"]]
                    image_name = prediction["image_name"]
                    pred_dir = os.path.join(output_dir, dataset_name)
                    pred_path = os.path.join(pred_dir, f"{image_name}.json")
                    ann_json = prediction["annotation"]
                    if not sly_fs.dir_exists(pred_dir):
                        sly_fs.mkdir(pred_dir)
                    sly_json.dump_json_file(ann_json, pred_path)

        def predict_dataset_id_by_args(
            api: Api,
            dataset_ids: List[int],
            output_dir: str = "./predictions",
            settings: str = None,
            draw: bool = False,
            upload: bool = False,
        ):
            if draw:
                raise ValueError("Draw visualization is not supported for dataset inference")
            if self.api is None:
                raise ValueError(missing_env_message)
            dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
            project_ids = list(set([dataset_info.project_id for dataset_info in dataset_infos]))
            if len(project_ids) > 1:
                raise ValueError("All datasets should belong to the same project")
            predict_project_id_by_args(
                api, project_ids[0], dataset_ids, output_dir, settings, draw, upload
            )

        def predict_image_id_by_args(
            api: Api,
            image_id: int,
            output_dir: str = "./predictions",
            settings: str = None,
            draw: bool = False,
            upload: bool = False,
        ):
            if self.api is None:
                raise ValueError(missing_env_message)

            logger.info(f"Predicting image: '{image_id}'")

            def predict_image_np(image_np):
                anns, _ = self._inference_auto([image_np], settings)
                if len(anns) == 0:
                    return Annotation(img_size=image_np.shape[:2])
                ann = anns[0]
                return ann

            image_np = api.image.download_np(image_id)
            ann = predict_image_np(image_np)

            image_info = None
            if not upload:
                ann_json = ann.to_json()
                image_info = api.image.get_info_by_id(image_id)
                dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
                pred_dir = os.path.join(output_dir, dataset_info.name)
                pred_path = os.path.join(pred_dir, f"{image_info.name}.json")
                if not sly_fs.dir_exists(pred_dir):
                    sly_fs.mkdir(pred_dir)
                sly_json.dump_json_file(ann_json, pred_path)

            if draw:
                if image_info is None:
                    image_info = api.image.get_info_by_id(image_id)
                vis_path = os.path.join(output_dir, dataset_info.name, f"{image_info.name}.png")
                ann.draw_pretty(image_np, output_path=vis_path)
            if upload:
                api.annotation.upload_ann(image_id, ann)

        def predict_local_data_by_args(
            input_path: str,
            settings: str = None,
            output_dir: str = "./predictions",
            draw: bool = False,
        ):
            logger.info(f"Predicting '{input_path}'")

            def postprocess_image(image_path: str, ann: Annotation, pred_dir: str = None):
                image_name = sly_fs.get_file_name_with_ext(image_path)
                if pred_dir is not None:
                    pred_dir = os.path.join(output_dir, pred_dir)
                    pred_ann_path = os.path.join(pred_dir, f"{image_name}.json")
                else:
                    pred_dir = output_dir
                    pred_ann_path = os.path.join(pred_dir, f"{image_name}.json")

                if not os.path.exists(pred_dir):
                    sly_fs.mkdir(pred_dir)
                sly_json.dump_json_file(ann.to_json(), pred_ann_path)
                if draw:
                    image = sly_image.read(image_path)
                    ann.draw_pretty(image, output_path=os.path.join(pred_dir, image_name))

            # 1. Input Directory
            if os.path.isdir(input_path):
                pred_dir = os.path.basename(input_path)
                images = list_files(input_path, valid_extensions=sly_image.SUPPORTED_IMG_EXTS)
                anns, _ = self._inference_auto(images, settings)
                for image_path, ann in zip(images, anns):
                    postprocess_image(image_path, ann, pred_dir)
            # 2. Input File
            elif os.path.isfile(input_path):
                if input_path.endswith(tuple(sly_image.SUPPORTED_IMG_EXTS)):
                    image_np = sly_image.read(input_path)
                    anns, _ = self._inference_auto([image_np], settings)
                    ann = anns[0]
                    postprocess_image(input_path, ann)
                elif input_path.endswith(tuple(ALLOWED_VIDEO_EXTENSIONS)):
                    raise NotImplementedError("Video inference is not implemented yet")
                else:
                    raise ValueError(
                        f"Unsupported input format: '{input_path}'. Expect image or directory with images"
                    )
            else:
                raise ValueError(f"Please provide a valid input path: '{input_path}'")

        if self._args.project_id is not None:
            predict_project_id_by_args(
                self.api,
                self._args.project_id,
                None,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.dataset_id is not None:
            predict_dataset_id_by_args(
                self.api,
                self._args.dataset_id,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.image_id is not None:
            predict_image_id_by_args(
                self.api,
                self._args.image_id,
                self._args.output,
                self._args.settings,
                self._args.draw,
                self._args.upload,
            )
        elif self._args.input is not None:
            predict_local_data_by_args(
                self._args.input,
                self._args.settings,
                self._args.output,
                self._args.draw,
            )

    def _add_workflow_input(self, model_source: str, model_files: dict, model_info: dict):
        if model_source == ModelSource.PRETRAINED:
            checkpoint_url = model_info["meta"]["model_files"]["checkpoint"]
            checkpoint_name = _get_model_name(model_info)
        else:
            checkpoint_name = sly_fs.get_file_name_with_ext(model_files["checkpoint"])
            checkpoint_url = os.path.join(
                model_info["artifacts_dir"], "checkpoints", checkpoint_name
            )

        app_name = sly_env.app_name()
        meta = WorkflowMeta(node_settings=WorkflowSettings(title=app_name))

        logger.debug(
            f"Workflow Input: Checkpoint URL - {checkpoint_url}, Checkpoint Name - {checkpoint_name}"
        )
        if model_source == ModelSource.CUSTOM:
            if checkpoint_url and self.api.file.exists(sly_env.team_id(), checkpoint_url):
                # self.api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
                remote_checkpoint_dir = os.path.dirname(checkpoint_url)
                self.api.app.workflow.add_input_folder(remote_checkpoint_dir, meta=meta)
            else:
                logger.debug(
                    f"Checkpoint {checkpoint_url} not found in Team Files. Cannot set workflow input"
                )


def _exclude_duplicated_predictions(
    api: Api,
    pred_anns: List[Annotation],
    dataset_id: int,
    gt_image_ids: List[int],
    iou: float = None,
    meta: Optional[ProjectMeta] = None,
):
    """
    Filter out predictions that significantly overlap with ground truth (GT) objects.

    This is a wrapper around the `_filter_duplicated_predictions_from_ann` method that does the following:
    - Checks inference settings for the IoU threshold (`existing_objects_iou_thresh`)
    - Gets ProjectMeta object if not provided
    - Downloads GT annotations for the specified image IDs
    - Filters out predictions that have an IoU greater than or equal to the specified threshold with any GT object

    :param api: Supervisely API object
    :type api: Api
    :param pred_anns: List of Annotation objects containing predictions
    :type pred_anns: List[Annotation]
    :param dataset_id: ID of the dataset containing the images
    :type dataset_id: int
    :param gt_image_ids: List of image IDs to filter predictions. All images should belong to the same dataset
    :type gt_image_ids: List[int]
    :param iou: IoU threshold (0.0-1.0). Predictions with IoU >= threshold with any
                    ground truth box of the same class will be removed. None if no filtering is needed
    :type iou: Optional[float]
    :param meta: ProjectMeta object
    :type meta: Optional[ProjectMeta]
    :return: List of Annotation objects containing filtered predictions
    :rtype: List[Annotation]

    Notes:
    ------
    - Requires PyTorch and torchvision for IoU calculations
    - This method is useful for identifying new objects that aren't already annotated in the ground truth
    """
    if isinstance(iou, float) and 0 < iou <= 1:
        if meta is None:
            ds = api.dataset.get_info_by_id(dataset_id)
            meta = ProjectMeta.from_json(api.project.get_meta(ds.project_id))
        gt_anns = api.annotation.download_json_batch(dataset_id, gt_image_ids)
        gt_anns = [Annotation.from_json(ann, meta) for ann in gt_anns]
        for i in range(0, len(pred_anns)):
            before = len(pred_anns[i].labels)
            with Timer() as timer:
                pred_anns[i] = _filter_duplicated_predictions_from_ann(
                    gt_anns[i], pred_anns[i], iou
                )
            after = len(pred_anns[i].labels)
            logger.debug(
                f"{[i]}: applied NMS with IoU={iou}. Before: {before}, After: {after}. Time: {timer.get_time():.3f}ms"
            )
    return pred_anns


def _filter_duplicated_predictions_from_ann(
    gt_ann: Annotation, pred_ann: Annotation, iou_threshold: float
) -> Annotation:
    """
    Filter out predictions that significantly overlap with ground truth annotations.

    This function compares each prediction with ground truth annotations of the same class
    and removes predictions that have an IoU (Intersection over Union) greater than or equal
    to the specified threshold with any ground truth annotation. This is useful for identifying
    new objects that aren't already annotated in the ground truth.

    :param gt_ann: Annotation object containing ground truth labels
    :type gt_ann: Annotation
    :param pred_ann: Annotation object containing prediction labels to be filtered
    :type pred_ann: Annotation
    :param iou_threshold:   IoU threshold (0.0-1.0). Predictions with IoU >= threshold with any
                            ground truth box of the same class will be removed
    :type iou_threshold: float
    :return: A new annotation object containing only predictions that don't significantly
                overlap with ground truth annotations
    :rtype: Annotation


    Notes:
    ------
    - Predictions with classes not present in ground truth will be kept
    - Requires PyTorch and torchvision for IoU calculations
    """

    try:
        import torch
        from torchvision.ops import box_iou

    except ImportError:
        raise ImportError("Please install PyTorch and torchvision to use this feature.")

    def _to_tensor(geom):
        return torch.tensor([geom.left, geom.top, geom.right, geom.bottom]).float()

    new_labels = []
    pred_cls_bboxes = defaultdict(list)
    for label in pred_ann.labels:
        pred_cls_bboxes[label.obj_class.name].append(label)

    gt_cls_bboxes = defaultdict(list)
    for label in gt_ann.labels:
        if label.obj_class.name not in pred_cls_bboxes:
            continue
        gt_cls_bboxes[label.obj_class.name].append(label)

    for name, pred in pred_cls_bboxes.items():
        gt = gt_cls_bboxes[name]
        if len(gt) == 0:
            new_labels.extend(pred)
            continue
        pred_bboxes = torch.stack([_to_tensor(l.geometry.to_bbox()) for l in pred]).float()
        gt_bboxes = torch.stack([_to_tensor(l.geometry.to_bbox()) for l in gt]).float()
        iou_matrix = box_iou(pred_bboxes, gt_bboxes)
        iou_matrix = iou_matrix.cpu().numpy()
        keep_indices = np.where(np.all(iou_matrix < iou_threshold, axis=1))[0]
        new_labels.extend([pred[i] for i in keep_indices])

    return pred_ann.clone(labels=new_labels)


def _get_log_extra_for_inference_request(
    inference_request_uuid, inference_request: Union[InferenceRequest, dict]
):
    if isinstance(inference_request, dict):
        log_extra = {
            "uuid": inference_request_uuid,
            "progress": inference_request["progress"],
            "is_inferring": inference_request["is_inferring"],
            "cancel_inference": inference_request["cancel_inference"],
            "has_result": inference_request["result"] is not None,
            "pending_results": len(inference_request["pending_results"]),
        }
        return log_extra

    progress = inference_request.progress_json()
    del progress["message"]
    log_extra = {
        "uuid": inference_request.uuid,
        "progress": progress,
        "is_inferring": inference_request.is_inferring(),
        "stopped": inference_request.is_stopped(),
        "finished": inference_request.is_finished(),
        "cancel_inference": inference_request.is_stopped(),
        "has_result": inference_request.final_result is not None,
        "pending_results": inference_request.pending_num(),
        "exception": inference_request.exception_json(),
        "result": inference_request._final_result,
        "preparing_progress": progress,
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


def progress_wrapper(func, progress_cb):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        progress_cb(len(result))
        return result

    return wrapped_func


def batched_iter(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_value_for_keys(data: dict, keys: List, ignore_none: bool = False):
    for key in keys:
        if key in data:
            if ignore_none and data[key] is None:
                continue
            return data[key]
    return None
