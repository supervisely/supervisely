# coding: utf-8
"""deploy pretrained and custom models"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import supervisely.io.env as env
from supervisely._utils import get_valid_kwargs
from supervisely.api.api import Api
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.experiments import ExperimentInfo
from supervisely.nn.utils import RuntimeType
from supervisely.sly_logger import logger


def get_runtime(runtime: Optional[str] = None):
    from supervisely.nn.utils import RuntimeType

    if runtime is None:
        return None
    aliases = {
        str(RuntimeType.PYTORCH): RuntimeType.PYTORCH,
        str(RuntimeType.ONNXRUNTIME): RuntimeType.ONNXRUNTIME,
        str(RuntimeType.TENSORRT): RuntimeType.TENSORRT,
        "pytorch": RuntimeType.PYTORCH,
        "torch": RuntimeType.PYTORCH,
        "pt": RuntimeType.PYTORCH,
        "onnxruntime": RuntimeType.ONNXRUNTIME,
        "onnx": RuntimeType.ONNXRUNTIME,
        "tensorrt": RuntimeType.TENSORRT,
        "trt": RuntimeType.TENSORRT,
        "engine": RuntimeType.TENSORRT,
    }
    if runtime in aliases:
        return aliases[runtime]
    runtime = aliases.get(runtime.lower(), None)
    if runtime is None:
        raise ValueError(
            f"Runtime '{runtime}' is not supported. Supported runtimes are: {', '.join(aliases.keys())}"
        )
    return runtime


class DeployApi:
    """ """

    def __init__(self, api: "Api"):
        self._api = api

    def load_pretrained_model(
        self,
        session_id: int,
        model_name: str,
        device: Optional[str] = None,
        runtime: str = None,
    ):
        """
        Load a pretrained model in running serving App.

        :param session_id: Task ID of the serving App.
        :type session_id: int
        :param model_name: Model name to deploy.
        :type model_name: str
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

        runtime = get_runtime(runtime)
        deploy_params = {}
        deploy_params["model_source"] = ModelSource.PRETRAINED
        deploy_params["device"] = device
        deploy_params["runtime"] = runtime
        self._load_model_from_api(session_id, deploy_params, model_name=model_name)

    def load_custom_model(
        self,
        session_id: int,
        team_id: int,
        artifacts_dir: str,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = None,
    ):
        """
        Load a custom model in running serving App.

        :param session_id: Task ID of the serving App.
        :type session_id: int
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param artifacts_dir: Path to the artifacts directory in the team fies.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, checkpoint will be chosen automatically, depending on the app version.
        :type checkpoint_name: Optional[str]
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

        runtime = get_runtime(runtime)

        # Train V1 logic (if artifacts_dir does not start with '/experiments')
        if not artifacts_dir.startswith("/experiments"):
            logger.debug("Deploying model from Train V1 artifacts")
            _, _, deploy_params = self._deploy_params_v1(
                team_id, artifacts_dir, checkpoint_name, device, runtime, with_module=False
            )
        else:  # Train V2 logic (when artifacts_dir starts with '/experiments')
            logger.debug("Deploying model from Train V2 artifacts")

            _, _, deploy_params = self._deploy_params_v2(
                team_id, artifacts_dir, checkpoint_name, device, runtime, with_module=False
            )
        deploy_params["model_source"] = ModelSource.CUSTOM
        self._load_model_from_api(session_id, deploy_params)

    def load_custom_model_from_experiment_info(
        self,
        session_id: int,
        experiment_info: "ExperimentInfo",
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = None,
    ):
        """
        Load a custom model in running serving App based on the training session.

        :param session_id: Task ID of the serving App.
        :type session_id: int
        :param experiment_info: an :class:`ExperimentInfo` object.
        :type experiment_info: ExperimentInfo
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, checkpoint will be chosen automatically, depending on the app version.
        :type checkpoint_name: Optional[str]
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

        runtime = get_runtime(runtime)
        if checkpoint_name is None:
            checkpoint_name = experiment_info.best_checkpoint
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {
                key: Path(experiment_info.artifacts_dir, value).as_posix()
                for key, value in experiment_info.model_files.items()
            },
            "model_info": experiment_info.to_json(),
            "runtime": runtime,
        }
        deploy_params["model_files"]["checkpoint"] = Path(
            experiment_info.artifacts_dir, "checkpoints", checkpoint_name
        ).as_posix()
        self._load_model_from_api(session_id, deploy_params)

    def _find_agent(self, team_id: int = None, public=True, gpu=True):
        """
        Find an agent in Supervisely with most available memory.

        :param team_id: Team ID. If not provided, will be taken from the current context.
        :type team_id: Optional[int]
        :param public: If True, can find a public agent.
        :type public: bool
        :param gpu: If True, find an agent with GPU.
        :type gpu: bool
        :return: Agent ID
        :rtype: int
        """
        if team_id is None:
            team_id = env.team_id()
        agents = self._api.agent.get_list_available(team_id, show_public=public, has_gpu=gpu)
        if len(agents) == 0:
            raise ValueError("No available agents found.")
        agent_id_memory_map = {}
        kubernetes_agents = []
        for agent in agents:
            if agent.type == "sly_agent":
                # No multi-gpu support, always take the first one
                agent_id_memory_map[agent.id] = agent.gpu_info["device_memory"][0]["available"]
            elif agent.type == "kubernetes":
                kubernetes_agents.append(agent.id)
        if len(agent_id_memory_map) > 0:
            return max(agent_id_memory_map, key=agent_id_memory_map.get)
        if len(kubernetes_agents) > 0:
            return kubernetes_agents[0]

    def deploy_pretrained_model(
        self,
        framework: Union[str, int],
        model_name: str,
        device: Optional[str] = None,
        runtime: str = None,
        workspace_id: int = None,
        agent_id: Optional[int] = None,
        app: Union[str, int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a pretrained model.

        :param framework: Framework name or Framework ID in Supervisely.
        :type framework: Union[str, int]
        :param model_name: Model name to deploy.
        :type model_name: str
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        :param workspace_id: Workspace ID where the app will be deployed. If not provided, will be taken from the current context.
        :type workspace_id: Optional[int]
        :param agent_id: Agent ID. If not provided, will be found automatically.
        :type agent_id: Optional[int]
        :param app: App name or App module ID in Supervisely.
        :type app: Union[str, int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if no serving apps found for the app name or multiple serving apps found for the app name.
        """
        from supervisely.nn.artifacts import (
            RITM,
            RTDETR,
            Detectron2,
            MMClassification,
            MMPretrain,
            MMDetection,
            MMDetection3,
            MMSegmentation,
            UNet,
            YOLOv5,
            YOLOv5v2,
            YOLOv8,
        )
        from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
        from supervisely.nn.utils import ModelSource

        workspace_info = self._api.workspace.get_info_by_id(workspace_id)
        if workspace_info is None:
            raise ValueError(f"Workspace with ID {workspace_id} not found")
        team_id = workspace_info.team_id

        # @TODO: Fix debug logs/ Fix code
        # Skip HTTPS redirect check on API init: False. ENV: False. Checked servers: set()
        frameworks_v1 = {
            RITM(team_id).framework_name: RITM(team_id).serve_slug,
            RTDETR(team_id).framework_name: RTDETR(team_id).serve_slug,
            Detectron2(team_id).framework_name: Detectron2(team_id).serve_slug,
            MMClassification(team_id).framework_name: MMClassification(team_id).serve_slug,
            MMPretrain(team_id).framework_name: MMPretrain(team_id).serve_slug,
            MMDetection(team_id).framework_name: MMDetection(team_id).serve_slug,
            MMDetection3(team_id).framework_name: MMDetection3(team_id).serve_slug,
            MMSegmentation(team_id).framework_name: MMSegmentation(team_id).serve_slug,
            UNet(team_id).framework_name: UNet(team_id).serve_slug,
            YOLOv5(team_id).framework_name: YOLOv5(team_id).serve_slug,
            YOLOv5v2(team_id).framework_name: YOLOv5v2(team_id).serve_slug,
            YOLOv8(team_id).framework_name: YOLOv8(team_id).serve_slug,
        }
        if framework in frameworks_v1:
            slug = frameworks_v1[framework]
            module_id = self.find_serving_app_by_slug(slug)
        else:
            module_id = None
            if isinstance(app, int):
                module_id = app
            elif isinstance(app, str):
                module_id = self._api.app.find_module_id_by_app_name(app)
            else:
                module_id = self.find_serving_app_by_framework(framework)["id"]
        if module_id is None:
            raise ValueError(
                f"Serving app for framework '{framework}' not found. Make sure that you used correct framework name."
            )

        runtime = get_runtime(runtime)
        if agent_id is None:
            agent_id = self._find_agent()

        task_info = self._run_serve_app(agent_id, module_id, workspace_id=workspace_id, **kwargs)
        self.load_pretrained_model(
            task_info["id"], model_name=model_name, device=device, runtime=runtime
        )
        return task_info

    def _find_team_by_path(self, path: str, team_id: int = None, raise_not_found=True):
        if team_id is not None:
            if self._api.file.exists(team_id, path) or self._api.file.dir_exists(
                team_id, path, recursive=False
            ):
                return team_id
            elif raise_not_found:
                raise ValueError(f"Checkpoint '{path}' not found in team provided team")
            else:
                return None
        team_id = env.team_id(raise_not_found=False)
        if team_id is not None:
            if self._api.file.exists(team_id, path) or self._api.file.dir_exists(
                team_id, path, recursive=False
            ):
                return team_id
        teams = self._api.team.get_list()
        team_id = None
        for team in teams:
            if self._api.file.exists(team.id, path):
                if team_id is not None:
                    raise ValueError("Multiple teams have the same checkpoint")
                team_id = team.id
        if team_id is None:
            if raise_not_found:
                raise ValueError("Checkpoint not found")
            else:
                return None
        return team_id

    def deploy_custom_model_by_checkpoint(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        runtime: str = None,
        timeout: int = 100,
        team_id: int = None,
        workspace_id: int = None,
        agent_id: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a custom model based on the checkpoint path.

        :param checkpoint: Path to the checkpoint in Team Files.
        :type checkpoint: str
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        :param timeout: Timeout in seconds (default is 100). The maximum time to wait for the serving app to be ready.
        :type timeout: Optional[int]
        :param team_id: Team ID where the artifacts are stored. If not provided, will be taken from the current context.
        :type team_id: Optional[int]
        :param workspace_id: Workspace ID where the app will be deployed. If not provided, will be taken from the current context.
        :type workspace_id: Optional[int]
        :param agent_id: Agent ID. If not provided, will be found automatically.
        :type agent_id: Optional[int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        artifacts_dir, checkpoint_name = self._get_artifacts_dir_and_checkpoint_name(checkpoint)
        return self.deploy_custom_model_by_artifacts_dir(
            artifacts_dir=artifacts_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            runtime=runtime,
            timeout=timeout,
            team_id=team_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            **kwargs,
        )

    def deploy_custom_model_by_artifacts_dir(
        self,
        artifacts_dir: str,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = None,
        timeout: int = 100,
        team_id: int = None,
        workspace_id: int = None,
        agent_id: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a custom model based on the artifacts directory.

        :param artifacts_dir: Path to the artifacts directory in the team fies.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, checkpoint will be chosen automatically, depending on the app version.
        :type checkpoint_name: Optional[str]
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        :param timeout: Timeout in seconds (default is 100). The maximum time to wait for the serving app to be ready.
        :type timeout: Optional[int]
        :param team_id: Team ID where the artifacts are stored. If not provided, will be taken from the current context.
        :type team_id: Optional[int]
        :param workspace_id: Workspace ID where the app will be deployed. If not provided, will be taken from the current context.
        :type workspace_id: Optional[int]
        :param agent_id: Agent ID. If not provided, will be found automatically.
        :type agent_id: Optional[int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        from supervisely.nn.utils import ModelSource

        if not isinstance(artifacts_dir, str) or not artifacts_dir.strip():
            raise ValueError("artifacts_dir must be a non-empty string.")

        runtime = get_runtime(runtime)
        if team_id is None:
            team_id = self._find_team_by_path(artifacts_dir, team_id=team_id)
        logger.debug(
            f"Starting custom model deployment. Team: {team_id}, Artifacts Dir: '{artifacts_dir}'"
        )
        if agent_id is None:
            agent_id = self._find_agent()

        # Train V1 logic (if artifacts_dir does not start with '/experiments')
        if not artifacts_dir.startswith("/experiments"):
            logger.debug("Deploying model from Train V1 artifacts")
            module_id, serve_app_name, deploy_params = self._deploy_params_v1(
                team_id, artifacts_dir, checkpoint_name, device, runtime, with_module=True
            )
        else:  # Train V2 logic (when artifacts_dir starts with '/experiments')
            logger.debug("Deploying model from Train V2 artifacts")

            module_id, serve_app_name, deploy_params = self._deploy_params_v2(
                team_id, artifacts_dir, checkpoint_name, device, runtime, with_module=True
            )
        deploy_params["model_source"] = ModelSource.CUSTOM

        logger.info(
            f"{serve_app_name} app deployment started. Checkpoint: '{checkpoint_name}'. Deploy params: '{deploy_params}'"
        )
        try:
            task_info = self._run_serve_app(
                agent_id, module_id, workspace_id=workspace_id, **kwargs
            )
            self._load_model_from_api(task_info["id"], deploy_params)
        except Exception as e:
            raise RuntimeError(f"Failed to run '{serve_app_name}': {e}") from e
        return task_info

    def deploy_custom_model_from_experiment_info(
        self,
        agent_id: int,
        experiment_info: "ExperimentInfo",
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = None,
        timeout: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a custom model based on the training session.

        :param experiment_info: an :class:`ExperimentInfo` object.
        :type experiment_info: ExperimentInfo
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, the best checkpoint will be chosen.
        :type checkpoint_name: Optional[str]
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param timeout: Timeout in seconds (default is 100). The maximum time to wait for the serving app to be ready.
        :type timeout: Optional[int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        task_id = experiment_info.task_id
        train_task_info = self._api.task.get_info_by_id(task_id)
        runtime = get_runtime(runtime)

        logger.debug(f"Starting model deployment from experiment info. Task ID: '{task_id}'")
        train_module_id = train_task_info["meta"]["app"]["moduleId"]
        module = self.get_serving_app_by_train_app(module_id=train_module_id)
        serve_app_name = module["name"]
        module_id = module["id"]
        logger.debug(f"Serving app detected: '{serve_app_name}'. Module ID: '{module_id}'")

        if checkpoint_name is None:
            checkpoint_name = experiment_info.best_checkpoint

        # Task parameters
        experiment_name = experiment_info.experiment_name
        task_name = experiment_name + f" ({checkpoint_name})"
        if "task_name" not in kwargs:
            kwargs["task_name"] = task_name

        description = f"""Serve from experiment
                Experiment name:   {experiment_name}
                Evaluation report: {experiment_info.evaluation_report_link}
            """
        while len(description) > 255:
            description = description.rsplit("\n", 1)[0]
        if "description" not in kwargs:
            kwargs["description"] = description

        logger.info(f"{serve_app_name} app deployment started. Checkpoint: '{checkpoint_name}'.")
        try:
            task_info = self._run_serve_app(agent_id, module_id, timeout=timeout, **kwargs)
            self.load_custom_model_from_experiment_info(
                task_info["id"], experiment_info, checkpoint_name, device, runtime
            )
        except Exception as e:
            raise RuntimeError(f"Failed to run '{serve_app_name}': {e}") from e
        return task_info

    def start_serve_app(
        self, agent_id: int, app_name=None, module_id=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Run a serving app. Either app_name or module_id must be provided.

        :param app_name: App name in Supervisely.
        :type app_name: Optional[str]
        :param module_id: Module ID in Supervisely.
        :type module_id: Optional[int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        """
        if app_name is None and module_id is None:
            raise ValueError("Either app_name or module_id must be provided.")
        if app_name is not None and module_id is not None:
            raise ValueError("Only one of app_name or module_id must be provided.")
        if module_id is None:
            module_id = self._api.app.find_module_id_by_app_name(app_name)
        self._run_serve_app(agent_id, module_id, **kwargs)

    def _run_serve_app(
        self, agent_id: int, module_id, workspace_id: int = None, timeout: int = 100, **kwargs
    ):
        _attempt_delay_sec = 1
        _attempts = timeout // _attempt_delay_sec

        if workspace_id is None:
            workspace_id = env.workspace_id()
        kwargs = get_valid_kwargs(
            kwargs=kwargs,
            func=self._api.task.start,
            exclude=["self", "module_id", "workspace_id", "agent_id"],
        )
        task_info = self._api.task.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=workspace_id,
            **kwargs,
        )
        ready = self._api.app.wait_until_ready_for_api_calls(
            task_info["id"], _attempts, _attempt_delay_sec
        )
        if not ready:
            raise TimeoutError(
                f"Task {task_info['id']} is not ready for API calls after {timeout} seconds."
            )
        return task_info

    def _load_model_from_api(self, task_id, deploy_params, model_name: Optional[str] = None):
        logger.info("Loading model")
        self._api.task.send_request(
            task_id,
            "deploy_from_api",
            data={"deploy_params": deploy_params, "model_name": model_name},
            raise_error=True,
        )
        logger.info("Model loaded successfully")

    def find_serving_app_by_framework(self, framework: str):
        modules = self._api.app.get_list_ecosystem_modules(
            categories=["serve", f"framework:{framework}"], categories_operation="and"
        )
        if len(modules) == 0:
            return None
        return modules[0]

    def find_serving_app_by_slug(self, slug: str) -> int:
        return self._api.app.get_ecosystem_module_id(slug)

    def get_serving_app_by_train_app(self, app_name: Optional[str] = None, module_id: int = None):
        if app_name is None and module_id is None:
            raise ValueError("Either app_name or module_id must be provided.")
        if app_name is not None:
            module_id = self._api.app.find_module_id_by_app_name(app_name)
        train_module_info = self._api.app.get_ecosystem_module_info(module_id)
        train_app_config = train_module_info.config
        categories = train_app_config["categories"]
        framework = None
        for category in categories:
            if category.lower().startswith("framework:"):
                framework = category.lstrip("framework:")
                break
        if framework is None:
            raise ValueError(
                "Unable to define serving app. Framework is not specified in the train app"
            )

        logger.debug(f"Detected framework: {framework}")
        module = self.find_serving_app_by_framework(framework)
        if module is None:
            raise ValueError(f"No serving apps found for framework {framework}")
        return module

    def get_deploy_info(self, task_id: int) -> Dict[str, Any]:
        """
        Get deploy info of a serving task.

        :param task_id: Task ID of the serving App.
        :type task_id: int
        :return: Deploy Info
        :rtype: Dict[str, Any]
        """
        return self._api.task.send_request(task_id, "get_deploy_info", data={}, raise_error=True)

    def _deploy_params_v1(
        self,
        team_id: int,
        artifacts_dir: str,
        checkpoint_name: str,
        device: str,
        runtime: str,
        with_module: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        from supervisely.nn.artifacts import RITM, YOLOv5
        from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
        from supervisely.nn.utils import ModelSource

        framework_cls = self._get_framework_by_path(artifacts_dir)
        if not framework_cls:
            raise ValueError(f"Unsupported framework for artifacts_dir: '{artifacts_dir}'")

        framework: BaseTrainArtifacts = framework_cls(team_id)
        if framework_cls is RITM or framework_cls is YOLOv5:
            raise ValueError(
                f"{framework.framework_name} framework is not supported for deployment"
            )

        runtime = get_runtime(runtime)
        logger.debug(f"Detected framework: '{framework.framework_name}'")
        module_id = None
        serve_app_name = None
        if with_module:
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
            logger.info("Checkpoint name not provided. Using the last checkpoint.")
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
            deploy_params["runtime"] = runtime
        return module_id, serve_app_name, deploy_params

    def _deploy_params_v2(
        self,
        team_id: int,
        artifacts_dir: str,
        checkpoint_name: str,
        device: str,
        runtime: Optional[str] = None,
        with_module: bool = True,
    ):
        from supervisely.nn.experiments import get_experiment_info_by_artifacts_dir
        from supervisely.nn.utils import ModelSource

        experiment_info = get_experiment_info_by_artifacts_dir(self._api, team_id, artifacts_dir)
        if not experiment_info:
            raise ValueError(
                f"Failed to retrieve experiment info for artifacts_dir: '{artifacts_dir}'"
            )

        runtime = get_runtime(runtime)
        module = None
        module_id = None
        serve_app_name = None
        if with_module:
            framework_name = experiment_info.framework_name
            module = self.find_serving_app_by_framework(framework_name)
            serve_app_name = module["name"]
            module_id = module["id"]
            logger.debug(f"Serving app detected: '{serve_app_name}'. Module ID: '{module_id}'")

        if len(experiment_info.checkpoints) == 0:
            raise ValueError(f"No checkpoints found in: '{artifacts_dir}'.")

        checkpoint = None
        if checkpoint_name is not None:
            if checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"):
                for checkpoint_path in experiment_info.checkpoints:
                    if get_file_name_with_ext(checkpoint_path) == checkpoint_name:
                        checkpoint = get_file_name_with_ext(checkpoint_path)
                        break
            elif checkpoint_name.endswith(".onnx"):
                checkpoint_path = experiment_info.export.get("ONNXRuntime")
                if checkpoint_path is None:
                    raise ValueError(f"ONNXRuntime export not found in: '{artifacts_dir}'.")
            elif checkpoint_name.endswith(".engine"):
                checkpoint_path = experiment_info.export.get("TensorRT")
                if checkpoint_path is None:
                    raise ValueError(f"TensorRT export not found in: '{artifacts_dir}'.")
            else:
                raise ValueError(
                    f"Unknown checkpoint format: '{checkpoint_name}'. Expected formats: '.pt', '.pth', '.onnx' or '.engine'"
                )

            checkpoint = get_file_name_with_ext(checkpoint_path)
            if checkpoint is None:
                raise ValueError(f"Provided checkpoint '{checkpoint_name}' not found")
        else:
            logger.info("Checkpoint name not provided. Using the best checkpoint.")
            checkpoint = experiment_info.best_checkpoint

        model_info_dict = asdict(experiment_info)
        model_info_dict["artifacts_dir"] = artifacts_dir
        checkpoint_name = checkpoint
        checkpoints_dir = self._get_checkpoints_dir(checkpoint_name)
        checkpoint_path = f"/{artifacts_dir.strip('/')}/{checkpoints_dir}/{checkpoint_name}"
        if runtime is None:
            runtime = self._set_auto_runtime_by_checkpoint(checkpoint_path)

        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {"checkpoint": checkpoint_path},
            "model_info": model_info_dict,
            "runtime": runtime,
        }

        for file_key, file_path in experiment_info.model_files.items():
            full_file_path = os.path.join(experiment_info.artifacts_dir, file_path)
            if not self._api.file.exists(team_id, full_file_path):
                logger.debug(
                    f"Model file not found: '{full_file_path}'. Trying to find it by checkpoint path."
                )
                full_file_path = os.path.join(artifacts_dir, file_path)
                if not self._api.file.exists(team_id, full_file_path):
                    raise ValueError(
                        f"Model file not found: '{full_file_path}'. Make sure that the file exists in the artifacts directory."
                    )
            deploy_params["model_files"][file_key] = full_file_path
            logger.debug(f"Model file added: {full_file_path}")
        return module_id, serve_app_name, deploy_params

    def _set_auto_runtime_by_checkpoint(self, checkpoint_path: str) -> str:
        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
            return RuntimeType.PYTORCH
        elif checkpoint_path.endswith(".onnx"):
            return RuntimeType.ONNXRUNTIME
        elif checkpoint_path.endswith(".engine"):
            return RuntimeType.TENSORRT
        else:
            raise ValueError(f"Unknown checkpoint format: '{checkpoint_path}'")

    def wait(self, model_id, target: Literal["started", "deployed"] = "started", timeout=5 * 60):
        t = time.monotonic()
        method = "is_alive" if target == "started" else "is_ready"
        while time.monotonic() - t < timeout:
            self._api.task.send_request(model_id, "is_ready", {})
            time.sleep(1)

    def _get_artifacts_dir_and_checkpoint_name(self, model: str) -> Tuple[str, str]:
        if not model.startswith("/"):
            raise ValueError(f"Path must start with '/'")

        if model.startswith("/experiments"):
            if model.endswith(".pt") or model.endswith(".pth"):
                try:
                    artifacts_dir, checkpoint_name = model.split("/checkpoints/")
                    return artifacts_dir, checkpoint_name
                except:
                    raise ValueError(
                        "Bad format of checkpoint path. Expected format: '/artifacts_dir/checkpoints/checkpoint_name'"
                    )
            elif model.endswith(".onnx") or model.endswith(".engine"):
                try:
                    artifacts_dir, checkpoint_name = model.split("/export/")
                    return artifacts_dir, checkpoint_name
                except:
                    raise ValueError(
                        "Bad format of checkpoint path. Expected format: '/artifacts_dir/export/checkpoint_name'"
                    )
            else:
                raise ValueError(f"Unknown model format: '{get_file_name_with_ext(model)}'")

        framework_cls = self._get_framework_by_path(model)
        if framework_cls is None:
            raise ValueError(f"Unknown path: '{model}'")

        team_id = env.team_id()
        framework = framework_cls(team_id)
        checkpoint_name = get_file_name_with_ext(model)
        checkpoints_dir = model.replace(checkpoint_name, "")
        if framework.weights_folder is not None:
            artifacts_dir = checkpoints_dir.replace(framework.weights_folder, "")
        else:
            artifacts_dir = checkpoints_dir
        return artifacts_dir, checkpoint_name

    def _get_checkpoints_dir(self, checkpoint_name: str) -> str:
        if checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"):
            return "checkpoints"
        elif checkpoint_name.endswith(".onnx") or checkpoint_name.endswith(".engine"):
            return "export"
        else:
            raise ValueError(f"Unknown checkpoint format: '{checkpoint_name}'")

    def _get_framework_by_path(self, path: str):
        from supervisely.nn.artifacts import (
            RITM,
            RTDETR,
            Detectron2,
            MMClassification,
            MMPretrain,
            MMDetection,
            MMDetection3,
            MMSegmentation,
            UNet,
            YOLOv5,
            YOLOv5v2,
            YOLOv8,
        )
        from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
        from supervisely.nn.utils import ModelSource

        path_obj = Path(path)
        if len(path_obj.parts) < 2:
            raise ValueError(f"Incorrect checkpoint path: '{path}'")
        parent = path_obj.parts[1]
        frameworks = {
            "/detectron2": Detectron2,
            "/mmclassification": MMClassification,
            "/mmclassification-v2": MMPretrain,
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
        if f"/{parent}" in frameworks:
            return frameworks[f"/{parent}"]
