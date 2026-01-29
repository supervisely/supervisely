# coding: utf-8
"""
Utilities for loading and running inference with deployed models.
"""

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
    """
    Client for interacting with a deployed model (load settings/metadata, run inference).

    The instance can be created either from a Supervisely Task ID (to resolve the deployment URL
    automatically) or from a direct deployment URL.

    :Task based usage:

        .. code-block:: python

            import os
            import supervisely as sly
            from supervisely.nn.model.model_api import ModelAPI

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            api = sly.Api.from_env()
            model = ModelAPI(api=api, task_id=12345)

            meta = model.get_model_meta()
            classes = model.get_classes()
            predictions = model.predict(image_id=100500, classes=classes)

    :Direct URL usage:

        .. code-block:: python

            from supervisely.nn.model.model_api import ModelAPI

            model = ModelAPI(url='https://app.supervisely.com/net/<sessionToken>')
            predictions = model.predict(input='/path/to/image.jpg')
    """

    def __init__(self, api: "Api" = None, task_id: int = None, url: str = None):
        """
        Create a deployed model client.

        Exactly one of ``task_id`` or ``url`` must be provided.

        :param api: API client. Required when ``task_id`` is used.
        :type api: :class:`~supervisely.api.api.Api`, optional
        :param task_id: Supervisely task id of a deployed model.
        :type task_id: int, optional
        :param url: Direct URL to a deployed model endpoint (e.g. ``https://.../net/<token>``).
        :type url: str, optional
        :raises AssertionError: If both ``task_id`` and ``url`` are provided, or neither is provided.
        :raises ValueError: If ``task_id`` is provided but the task is not found.
        """
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
        """
        Return deployment info for the current model.

        For task-based mode this calls internal deploy API, for URL-based mode it calls
        ``get_deploy_info`` endpoint of the deployment.

        :returns: Deployment info (raw JSON returned by the backend).
        :rtype: dict
        """
        if self.task_id is not None:
            return self.api.nn._deploy_api.get_deploy_info(self.task_id)
        return self._post("get_deploy_info", {})

    def get_settings(self):
        """
        Return custom inference settings for the deployed model.

        :returns: Settings dict.
        :rtype: dict
        """
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_custom_inference_settings", {})[
                "settings"
            ]
        else:
            return self._post("get_custom_inference_settings", {})["settings"]

    def get_tracking_settings(self):
        """
        Return tracking settings for the deployed model.

        Currently returns settings for the ``botsort`` tracker.

        :returns: Tracking settings dict.
        :rtype: dict
        """
        # @TODO: botsort hardcoded
        # Add dropdown selector for tracking algorithms later
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_tracking_settings", {})["botsort"]
        else:
            return self._post("get_tracking_settings", {})["botsort"]

    def get_model_meta(self):
        """
        Return output :class:`~supervisely.project.project_meta.ProjectMeta` for the deployed model.

        The meta typically includes object classes and tags that the model predicts.

        :returns: Model output meta.
        :rtype: :class:`~supervisely.project.project_meta.ProjectMeta`
        """
        if self.task_id is not None:
            return ProjectMeta.from_json(
                self.api.task.send_request(self.task_id, "get_output_classes_and_tags", {})
            )
        else:
            return ProjectMeta.from_json(self._post("get_output_classes_and_tags", {}))

    def get_classes(self):
        """
        Convenience wrapper to return output class names from :meth:`~.get_model_meta`.

        :returns: List of class names.
        :rtype: List[str]
        """
        model_meta = self.get_model_meta()
        return [obj_class.name for obj_class in model_meta.obj_classes]

    def list_pretrained_models(self) -> List[str]:
        """
        Return a list of pretrained model names available for deployment.

        :returns: Pretrained model names.
        :rtype: List[str]
        """
        return self._post("list_pretrained_models", {})

    def list_pretrained_model_infos(self) -> List[dict]:
        """
        Return a list of pretrained model infos with full information about each model.

        :returns: List of model info dicts.
        :rtype: List[dict]
        """
        return self._post("list_pretrained_model_infos", {})

    def list_experiments(self) -> List[ExperimentInfo]:
        """
        Return a list of training experiments in Supervisely.

        .. note::
            This method is not implemented.

        :returns: Experiments list.
        :rtype: List[:class:`~supervisely.nn.experiments.ExperimentInfo`]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError

    def is_deployed(self) -> bool:
        """
        Check whether the deployment is ready.

        :returns: True if ready.
        :rtype: bool
        """
        if self.task_id is not None:
            return self.api.task.is_ready(self.task_id)
        return self._post("is_ready", {})["status"] == "ready"

    def status(self):
        """
        Return deployment status JSON.

        :returns: Status dict.
        :rtype: dict
        """
        if self.task_id is not None:
            return self.api.task.send_request(self.task_id, "get_status", {})
        return self._post("get_status", {})

    def shutdown(self):
        """
        Stop the deployment task (task-based mode) or request shutdown (URL-based mode).

        :returns: Status info. In task-based mode returns task stop response, in URL-based mode returns
            status enum value if supported by the backend.
        """
        if self.task_id is not None:
            return self.api.task.stop(self.task_id)
        response = self._post("tasks.stop", {ApiField.ID: id})
        return TaskApi.Status(response[ApiField.STATUS])

    def freeze_model(self):
        """
        Freeze the model to free up resources.

        :returns: Backend response.
        :rtype: dict
        """
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
        """
        Load a model into the deployment.

        Behavior depends on the connection mode:

        - **URL-based Mode** (``task_id`` is None): if ``model`` points to an existing local file,
          it is treated as a custom checkpoint; otherwise it is treated as a pretrained model name.
        - **Task-based Mode**: if ``model`` starts with ``/`` it is treated as a path to a custom
          checkpoint in team files; otherwise it is treated as a pretrained model name.

        :param model: Pretrained model name or checkpoint path (depending on the mode).
        :type model: str
        :param device: Optional device spec (passed to deploy backend).
        :type device: str, optional
        :param runtime: Optional runtime spec (passed to deploy backend).
        :type runtime: str, optional
        :returns: Backend response (URL-based mode) or None (task-based mode).
        :rtype: dict or None
        :raises ValueError: If pretrained model name is not found (URL-based mode).
        """
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
        """
        Create a prediction session (lazy iterator).

        Use this method when you want to iterate predictions as they are produced, or if you need
        direct access to :class:`~supervisely.nn.model.prediction_session.PredictionSession`.

        Parameters are forwarded to :class:`~supervisely.nn.model.prediction_session.PredictionSession`.

        :returns: Prediction session.
        :rtype: :class:`~supervisely.nn.model.prediction_session.PredictionSession`
        """
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
        """
        Run inference and return predictions as a list.

        This is a convenience wrapper over :meth:`~.predict_detached` that consumes the session and
        returns ``list(session)``.

        Parameters are forwarded to :class:`~supervisely.nn.model.prediction_session.PredictionSession`.

        :returns: Predictions list.
        :rtype: List[:class:`~supervisely.nn.model.prediction.Prediction`]
        """
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
