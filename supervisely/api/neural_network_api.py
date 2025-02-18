# coding: utf-8
"""download/upload/manipulate neural networks"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from supervisely.api.module_api import CloneableModuleApi, RemoveableModuleApi
from supervisely.io.fs import get_file_name_with_ext
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api


class DeployModel:

    def __init__(self, api: "Api", timeout: int = 100):
        self._api = api
        self._timeout = timeout

    def pretrained(
        self,
        module_id: int,
        model_name: str,
        device: str = "cuda",
        runtime: str = "PyTorch",
        **kwargs,
    ):
        # TODO: from train_app and from train_module
        # TODO: custom from train module? with any url checkpoint
        from supervisely.nn.utils import ModelSource

        task_info = self._run_serve_app(module_id, **kwargs)
        deploy_info = self._get_deploy_info(task_info["id"])
        deploy_params = deploy_info["deploy_params"]
        deploy_params["model_source"] = ModelSource.PRETRAINED
        deploy_params["device"] = device
        deploy_params["runtime"] = runtime
        model_name  # <- separate arg for deploy_from_api request
        deploy_params[
            "model_info"
        ]  # <- to be defined in the inference class on deploy_params request by model name
        self._deploy_model_from_api(task_info["id"], deploy_params)

    def custom(
        self,
        module_id: int,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, Any]:
        from supervisely.nn.utils import ModelSource, RuntimeType

        task_info = self._run_serve_app(module_id, **kwargs)

        model_files = {"checkpoint": checkpoint_path}
        if config_path is not None:
            model_files["config"] = config_path
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": model_files,
            "model_info": {},  # what arguments are required here?
            "runtime": RuntimeType.PYTORCH,
        }
        self._deploy_model_from_api(task_info["id"], deploy_params)
        return task_info

    def from_artifacts(
        self,
        workspace_id: int,
        artifacts_dir: str,
        checkpoint_name: str = None,
        device: str = "cuda",
        timeout: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a custom model based on the artifacts directory.

        :param workspace_id: Workspace ID in Supervisely.
        :type workspace_id: int
        :param artifacts_dir: Path to the artifacts directory.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name (with extension) to deploy.
        :type checkpoint_name: Optional[str]
        :param device: Device string (default is "cuda").
        :type device: str
        :param timeout: Timeout in seconds (default is 100).
        :type timeout: int
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        if not isinstance(workspace_id, int) or workspace_id <= 0:
            raise ValueError(f"workspace_id must be a positive integer. Received: {workspace_id}")
        if not isinstance(artifacts_dir, str) or not artifacts_dir.strip():
            raise ValueError("artifacts_dir must be a non-empty string.")

        workspace_info = self._api.workspace.get_info_by_id(workspace_id)
        if workspace_info is None:
            raise ValueError(f"Workspace with ID '{workspace_id}' not found.")

        team_id = workspace_info.team_id
        logger.debug(
            f"Starting model deployment. Team: {team_id}, Workspace: {workspace_id}, Artifacts Dir: '{artifacts_dir}'"
        )

        # Train V1 logic (if artifacts_dir does not start with '/experiments')
        if not artifacts_dir.startswith("/experiments"):
            logger.debug("Deploying model from Train V1 artifacts")
            module_id, serve_app_name, deploy_params = self.__deploy_params_v1(
                team_id, artifacts_dir, checkpoint_name, device
            )
        else:  # Train V2 logic (when artifacts_dir starts with '/experiments')
            logger.debug("Deploying model from Train V2 artifacts")

            module_id, serve_app_name, deploy_params = self.__deploy_params_v2(
                team_id, artifacts_dir, checkpoint_name, device
            )

        if "workspace_id" not in kwargs:
            kwargs["workspace_id"] = workspace_id

        logger.info(
            f"{serve_app_name} app deployment started. Checkpoint: '{checkpoint_name}'. Deploy params: '{deploy_params}'"
        )
        try:
            task_info = self._run_serve_app(module_id, timeout=timeout, **kwargs)
            self._deploy_model_from_api(task_info["id"], deploy_params)
        except Exception as e:
            raise RuntimeError(f"Failed to run '{serve_app_name}': {e}") from e
        return task_info

    def from_train_task(
        self,
        task_id: int,
        checkpoint_name: str = None,
        device: str = "cuda",
        timeout: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a custom model based on the training task.

        :param task_id: Task ID of Train App in Supervisely.
        :type task_id: int
        :param checkpoint_name: Checkpoint name (with extension) to deploy.
        :type checkpoint_name: Optional[str]
        :param device: Device string (default is "cuda").
        :type device: str
        :param timeout: Timeout in seconds (default is 100).
        :type timeout: int
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        from supervisely.nn.utils import ModelSource, RuntimeType

        task_info = self._api.task.get_info_by_id(task_id)
        try:
            data = task_info["meta"]["output"]["experiment"]["data"]
        except KeyError:
            raise ValueError("Task output does not contain experiment data")

        workspace_id = task_info["workspaceId"]
        workspace_info = self._api.workspace.get_info_by_id(workspace_id)
        if workspace_info is None:
            raise ValueError(f"Workspace with ID '{workspace_id}' not found.")

        team_id = workspace_info.team_id
        logger.debug(
            f"Starting model deployment. Team: {team_id}, Workspace: {workspace_id}, Task ID: '{task_id}'"
        )
        train_module_id = task_info["meta"]["app"]["moduleId"]
        module = self._get_serving_from_train(train_module_id)
        serve_app_name = module["name"]
        module_id = module["id"]
        logger.debug(f"Serving app detected: '{serve_app_name}'. Module ID: '{module_id}'")

        experiment_name = data["experiment_name"]
        if checkpoint_name is None:
            checkpoint_name = data["best_checkpoint"]

        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {
                "checkpoint": Path(
                    data["artifacts_dir"], "checkpoints", checkpoint_name
                ).as_posix(),
                "config": Path(data["artifacts_dir"], data["model_files"]["config"]).as_posix(),
            },
            "model_info": data,
            "runtime": RuntimeType.PYTORCH,
        }

        # Task parameters
        if "workspace_id" not in kwargs:
            kwargs["workspace_id"] = workspace_id
        task_name = experiment_name + f" ({checkpoint_name})"
        if "task_name" not in kwargs:
            kwargs["task_name"] = task_name

        description = f"""Serve from experiment
                Experiment name:   {experiment_name}
                Evaluation report: {data["evaluation_report_link"]}
            """
        while len(description) > 255:
            description = description.rsplit("\n", 1)[0]
        if "description" not in kwargs:
            kwargs["description"] = description

        logger.info(
            f"{serve_app_name} app deployment started. Checkpoint: '{checkpoint_name}'. Deploy params: '{deploy_params}'"
        )
        try:
            task_info = self._run_serve_app(module_id, timeout=timeout, **kwargs)
            self._deploy_model_from_api(task_info["id"], deploy_params)
        except Exception as e:
            raise RuntimeError(f"Failed to run '{serve_app_name}': {e}") from e
        return task_info

    def _run_serve_app(self, module_id, timeout: int = None, **kwargs):
        if timeout is None:
            timeout = self._timeout
        _attempt_delay_sec = 1
        _attempts = timeout // _attempt_delay_sec

        task_info = self._api.task.start(module_id=module_id, **kwargs)
        ready = self._api.app.wait_until_ready_for_api_calls(
            task_info["id"], _attempts, _attempt_delay_sec
        )
        if not ready:
            raise TimeoutError(
                f"Task {task_info['id']} is not ready for API calls after {self._timeout} seconds."
            )
        return task_info

    def _deploy_model_from_api(self, task_id, deploy_params):
        self._api.task.send_request(
            task_id, "deploy_from_api", data={"deploy_params": deploy_params}, raise_error=True
        )

    def _get_deploy_info(self, task_id: int):
        return self._api.task.send_request(task_id, "get_deploy_info", raise_error=True)

    def _get_serving_from_train(self, train_app_module_id: int):
        train_module_info = self._api.app.get_ecosystem_module_info(train_app_module_id)
        train_app_config = train_module_info.config
        categories = train_app_config["categories"]
        framework = None
        for category in categories:
            if category.lower().startswith("framework:"):
                framework = category
                break
        if framework is None:
            raise ValueError(
                "Unable to define serving app. Framework is not specified in the train app"
            )

        logger.debug(f"Detected framework: {framework}")
        modules = self._api.app.get_list_ecosystem_modules(
            categories=["serve", framework], categories_operation="and"
        )
        if len(modules) == 0:
            raise ValueError(f"No serving apps found for framework {framework}")
        return modules[0]

    def __deploy_params_v1(
        self, team_id: int, artifacts_dir: str, checkpoint_name: str, device: str
    ) -> Tuple[int, Dict[str, Any]]:
        from supervisely.nn.artifacts import (
            RITM,
            RTDETR,
            Detectron2,
            MMClassification,
            MMDetection,
            MMDetection3,
            MMSegmentation,
            UNet,
            YOLOv5,
            YOLOv5v2,
            YOLOv8,
        )
        from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
        from supervisely.nn.utils import ModelSource, RuntimeType

        frameworks = {
            "/detectron2": Detectron2,
            "/mmclassification": MMClassification,
            "/mmdetection": MMDetection,
            "/mmdetection-3": MMDetection3,
            "/mmsegmentation": MMSegmentation,
            "/RITM_training": RITM,
            "/RT-DETR": RTDETR,
            "/unet": UNet,
            "/yolov5_train": YOLOv5,
            "/yolov5_2.0_train": YOLOv5v2,
            "/yolov8_train": YOLOv8,
        }

        framework_cls = next(
            (cls for prefix, cls in frameworks.items() if artifacts_dir.startswith(prefix)),
            None,
        )
        if not framework_cls:
            raise ValueError(f"Unsupported framework for artifacts_dir: '{artifacts_dir}'")

        framework: BaseTrainArtifacts = framework_cls(team_id)
        if framework_cls is RITM or framework_cls is YOLOv5:
            raise ValueError(
                f"{framework.framework_name} framework is not supported for deployment"
            )

        logger.debug(f"Detected framework: '{framework.framework_name}'")

        module_id = self._api.app.get_ecosystem_module_id(framework.serve_slug)
        serve_app_name = framework.serve_app_name
        logger.debug(f"Module ID fetched:' {module_id}'. App name: '{serve_app_name}'")

        train_info = framework.get_info_by_artifacts_dir(artifacts_dir.rstrip("/"))
        if not hasattr(train_info, "checkpoints") or not train_info.checkpoints:
            raise ValueError("No checkpoints found in train info.")

        checkpoint = None
        if checkpoint_name is not None:
            for cp in train_info.checkpoints:
                if cp.name == checkpoint_name:
                    checkpoint = cp
                    break
            if checkpoint is None:
                raise ValueError(f"Checkpoint '{checkpoint_name}' not found in train info.")
        else:
            logger.debug("Checkpoint name not provided. Using the last checkpoint.")
            checkpoint = train_info.checkpoints[-1]

        checkpoint_name = checkpoint.name
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "task_type": train_info.task_type,
            "checkpoint_name": checkpoint_name,
            "checkpoint_url": checkpoint.path,
        }

        if getattr(train_info, "config_path", None) is not None:
            deploy_params["config_url"] = train_info.config_path

        if framework.require_runtime:
            deploy_params["runtime"] = RuntimeType.PYTORCH
        return module_id, serve_app_name, deploy_params

    def __deploy_params_v2(
        self, team_id: int, artifacts_dir: str, checkpoint_name: str, device: str
    ):
        from dataclasses import asdict

        from supervisely.nn.experiments import get_experiment_info_by_artifacts_dir
        from supervisely.nn.utils import ModelSource, RuntimeType

        experiment_info = get_experiment_info_by_artifacts_dir(self._api, team_id, artifacts_dir)
        if not experiment_info:
            raise ValueError(
                f"Failed to retrieve experiment info for artifacts_dir: '{artifacts_dir}'"
            )

        experiment_task_id = experiment_info.task_id
        experiment_task_info = self._api.task.get_info_by_id(experiment_task_id)
        if experiment_task_info is None:
            raise ValueError(f"Task with ID '{experiment_task_id}' not found")

        train_module_id = experiment_task_info["meta"]["app"]["moduleId"]
        module = self._get_serving_from_train(train_module_id)
        serve_app_name = module["name"]
        module_id = module["id"]
        logger.debug(f"Serving app detected: '{serve_app_name}'. Module ID: '{module_id}'")

        if len(experiment_info.checkpoints) == 0:
            raise ValueError(f"No checkpoints found in: '{artifacts_dir}'.")

        checkpoint = None
        if checkpoint_name is not None:
            for checkpoint_path in experiment_info.checkpoints:
                if get_file_name_with_ext(checkpoint_path) == checkpoint_name:
                    checkpoint = get_file_name_with_ext(checkpoint_path)
                    break
            if checkpoint is None:
                raise ValueError(f"Provided checkpoint '{checkpoint_name}' not found")
        else:
            logger.debug("Checkpoint name not provided. Using the best checkpoint.")
            checkpoint = experiment_info.best_checkpoint

        checkpoint_name = checkpoint
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {
                "checkpoint": f"{experiment_info.artifacts_dir}checkpoints/{checkpoint_name}"
            },
            "model_info": asdict(experiment_info),
            "runtime": RuntimeType.PYTORCH,
        }
        # TODO: add support for **kwargs

        config = experiment_info.model_files.get("config")
        if config is not None:
            deploy_params["model_files"]["config"] = f"{experiment_info.artifacts_dir}{config}"
            logger.debug(f"Config file added: {experiment_info.artifacts_dir}{config}")
        return module_id, serve_app_name, deploy_params


class NeuralNetworkApi(CloneableModuleApi, RemoveableModuleApi):
    """ """

    def __init__(self, api: "Api"):
        super().__init__(api)
        self.deploy = DeployModel(api)
