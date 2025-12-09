# coding: utf-8
"""load and inference models"""

from __future__ import annotations

import os
from os import PathLike
from typing import List, Union

import numpy as np
import requests

import supervisely.io.env as sly_env
import supervisely.io.json as sly_json
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.task_api import TaskApi
from supervisely.nn.experiments import ExperimentInfo
from supervisely.nn.model.prediction import Prediction
from supervisely.nn.model.prediction_session import PredictionSession
from supervisely.nn.utils import ModelSource
from supervisely.project.project_meta import ProjectMeta


class ModelAPI:
    def __init__(self, api: "Api" = None, task_id: int = None, url: str = None):
        assert not (task_id is None and url is None), "Either `task_id` or `url` must be passed."
        assert (
            task_id is None or url is None
        ), "Either `task_id` or `url` must be passed (not both)."
        if task_id is not None:
            assert api is not None, "API must be provided if `task_id` is passed."

        self.api = api
        self.task_id = task_id
        self.url = url

        if self.task_id is not None:
            task_info = self.api.task.get_info_by_id(self.task_id)
            if task_info is None:
                raise ValueError(f"Task with id {self.task_id} not found.")
            self.url = f'{self.api.server_address}/net/{task_info["meta"]["sessionToken"]}'

    # region Requests
    def _post(self, method: str, data: dict, raise_for_status: bool = True):
        url = f"{self.url.rstrip('/')}/{method.lstrip('/')}"
        response = requests.post(url, json=data)
        if raise_for_status:
            response.raise_for_status()
        return response.json()

    def _get(self, method: str, params: dict = None, raise_for_status: bool = True):
        url = f"{self.url.rstrip('/')}/{method.lstrip('/')}"
        response = requests.get(url, params=params)
        if raise_for_status:
            response.raise_for_status()
        return response.json()

    # ------------------------------------ #

    # region Info
    def get_info(self):
        if self.task_id is not None:
            return self.api.nn._deploy_api.get_deploy_info(self.task_id)
        return self._post("get_deploy_info", {})

    def get_settings(self):
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_custom_inference_settings", {})[
                "settings"
            ]
        else:
            return self._post("get_custom_inference_settings", {})["settings"]

    def get_tracking_settings(self):
        # @TODO: botsort hardcoded 
        # Add dropdown selector for tracking algorithms later
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_tracking_settings", {})["botsort"]
        else:
            return self._post("get_tracking_settings", {})["botsort"]


    def get_model_meta(self):
        if self.task_id is not None:
            return ProjectMeta.from_json(
                self.api.task.send_request(self.task_id, "get_output_classes_and_tags", {})
            )
        else:
            return ProjectMeta.from_json(self._post("get_output_classes_and_tags", {}))

    def get_classes(self):
        model_meta = self.get_model_meta()
        return [obj_class.name for obj_class in model_meta.obj_classes]

    def list_pretrained_models(self) -> List[str]:
        """Return a list of pretrained model names available for deployment"""
        return self._post("list_pretrained_models", {})

    def list_pretrained_model_infos(self) -> List[dict]:
        """Return a list of pretrained model infos with full information about each model"""
        return self._post("list_pretrained_model_infos", {})

    def list_experiments(self) -> List[ExperimentInfo]:
        """Return a list of training experiments in Supervisely"""
        raise NotImplementedError

    def is_deployed(self) -> bool:
        if self.task_id is not None:
            return self.api.task.is_ready(self.task_id)
        return self._post("is_ready", {})["status"] == "ready"

    def status(self):
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_status", {})
        return self._post("get_status", {})

    def shutdown(self):
        if self.task_id is not None:
            return self.api.task.stop(self.task_id)
        response = self._post("tasks.stop", {ApiField.ID: id})
        return TaskApi.Status(response[ApiField.STATUS])

    def freeze_model(self):
        """Freeze the model to free up resources."""
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "freeze_model", {})
        return self._post("freeze_model", {})

    # --------------------- #

    # region Load
    def load(
        self,
        model: str,
        device: str = None,
        runtime: str = None,
    ):
        if self.task_id is None:
            # TODO: proper check
            if os.path.exists(model):
                self._load_local_custom_model(model, device, runtime)
            else:
                self._load_local_pretrained_model(model, device, runtime)

        elif model.startswith("/"):
            team_id = sly_env.team_id()
            artifacts_dir, checkpoint_name = (
                self.api.nn._deploy_api._get_artifacts_dir_and_checkpoint_name(model)
            )
            self.api.nn._deploy_api.load_custom_model(
                self.task_id,
                team_id,
                artifacts_dir,
                checkpoint_name,
                device=device,
                runtime=runtime,
            )
        else:
            self.api.nn._deploy_api.load_pretrained_model(
                self.task_id, model, device=device, runtime=runtime
            )

    def _load_local_pretrained_model(self, model: str, device: str = None, runtime: str = None):
        available_models = self.list_pretrained_models()
        if model not in available_models:
            raise ValueError(f"Model {model} not found in available models: {available_models}")

        deploy_params = {
            "model_files": {},
            "model_source": ModelSource.PRETRAINED,
            "model_info": {},
            "device": device,
            "runtime": runtime,
        }
        state = {"deploy_params": deploy_params, "model_name": model}
        return self._post("deploy_from_api", {"state": state})

    def _load_local_custom_model(self, model: str, device: str = None, runtime: str = None):
        deploy_params = self._get_custom_model_params(model, device, runtime)
        state = {"deploy_params": deploy_params, "model_name": model}
        return self._post("deploy_from_api", {"state": state})

    def _get_custom_model_params(self, model_name: str, device: str = None, runtime: str = None):
        def _load_experiment_info(artifacts_dir):
            experiment_path = os.path.join(artifacts_dir, "experiment_info.json")
            model_info = sly_json.load_json_file(experiment_path)
            model_meta_path = os.path.join(artifacts_dir, "model_meta.json")
            model_info["model_meta"] = sly_json.load_json_file(model_meta_path)
            original_model_files = model_info.get("model_files")
            return model_info, original_model_files

        def _prepare_local_model_files(artifacts_dir, checkpoint_path, original_model_files):
            return {k: os.path.join(artifacts_dir, v) for k, v in original_model_files.items()} | {
                "checkpoint": checkpoint_path
            }

        model_source = ModelSource.CUSTOM
        artifacts_dir = os.path.dirname(os.path.dirname(model_name))
        model_info, original_model_files = _load_experiment_info(artifacts_dir)
        model_files = _prepare_local_model_files(artifacts_dir, model_name, original_model_files)
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
        }
        return deploy_params

    # --------------------------------- #

    # region Predict
    def predict_detached(
        self,
        input: Union[np.ndarray, str, PathLike, list] = None,
        image_id: int = None,
        video_id: int = None,
        dataset_id: int = None,
        project_id: int = None,
        batch_size: int = None,
        conf: float = None,
        img_size: int = None,
        classes: List[str] = None,
        upload_mode: str = None,
        recursive: bool = False,
        tracking: bool = None,
        tracking_config: dict = None,
        **kwargs,
    ) -> PredictionSession:
        
        return PredictionSession(
            self.url,
            input=input,
            image_id=image_id,
            video_id=video_id,
            dataset_id=dataset_id,
            project_id=project_id,
            api=self.api,
            batch_size=batch_size,
            conf=conf,
            img_size=img_size,
            classes=classes,
            upload_mode=upload_mode,
            recursive=recursive,
            tracking=tracking,
            tracking_config=tracking_config,
            **kwargs,
        )

    def predict(
        self,
        input: Union[np.ndarray, str, PathLike, list] = None,
        image_id: Union[List[int], int] = None,
        video_id: Union[List[int], int] = None,
        dataset_id: Union[List[int], int] = None,
        project_id: Union[List[int], int] = None,
        batch_size: int = None,
        conf: float = None,
        img_size: int = None,
        classes: List[str] = None,
        upload_mode: str = None,
        recursive: bool = False,
        tracking: bool = None,
        tracking_config: dict = None,
        **kwargs,
    ) -> List[Prediction]:
        if "show_progress" not in kwargs:
            kwargs["show_progress"] = True
        session = PredictionSession(
            self.url,
            input=input,
            image_id=image_id,
            video_id=video_id,
            dataset_id=dataset_id,
            project_id=project_id,
            api=self.api,
            batch_size=batch_size,
            conf=conf,
            img_size=img_size,
            classes=classes,
            upload_mode=upload_mode,
            recursive=recursive,
            tracking=tracking,
            tracking_config=tracking_config,
            **kwargs,
        )
        return list(session)

    # ------------------------------------ #
