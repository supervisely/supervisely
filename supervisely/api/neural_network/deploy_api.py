from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import supervisely.io.env as env
from supervisely.io.fs import get_file_name_with_ext
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo
    from supervisely.nn.utils import ModelSource, RuntimeType


class DeployApi:
    """ """

    def __init__(self, api: "Api"):
        self._api = api

    def load_pretrained_model(
        self,
        session_id: int,
        model_name: str,
        device: Optional[str] = None,
        runtime: Optional[str] = "RuntimeType.PYTORCH",
    ):
        """
        Load a pretrained model in running serving App.

        :param session_id: Task ID of the serving App.
        :type session_id: int
        :param model_name: Model name to deploy.
        :type model_name: str
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

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
        runtime: Optional[str] = "RuntimeType.PYTORCH",
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
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

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
        runtime: Optional[str] = "RuntimeType.PYTORCH",
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
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        """
        from supervisely.nn.utils import ModelSource

        if checkpoint_name is None:
            checkpoint_name = experiment_info.best_checkpoint
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {
                "checkpoint": Path(
                    experiment_info.artifacts_dir, "checkpoints", checkpoint_name
                ).as_posix(),
                "config": Path(
                    experiment_info.artifacts_dir, experiment_info.model_files["config"]
                ).as_posix(),
            },
            "model_info": experiment_info.to_json(),
            "runtime": runtime,
        }
        self._load_model_from_api(session_id, deploy_params)

    def deploy_pretrained_model(
        self,
        agent_id: int,
        app: Union[str, int],
        model_name: str,
        device: Optional[str] = None,
        runtime: Optional[str] = "RuntimeType.PYTORCH",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Deploy a pretrained model.

        :param app: App name or App module ID in Supervisely.
        :type app: Union[str, int]
        :param model_name: Model name to deploy.
        :type model_name: str
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if no serving apps found for the app name or multiple serving apps found for the app name.
        """
        if isinstance(app, int):
            module_id = app
        else:
            module_id = self._api.app.find_module_id_by_app_name(app)
        task_info = self._run_serve_app(agent_id, module_id, **kwargs)
        self.load_pretrained_model(
            task_info["id"], model_name=model_name, device=device, runtime=runtime
        )
        return task_info

    def deploy_custom_model_by_artifacts_dir(
        self,
        agent_id: int,
        artifacts_dir: str,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime: Optional[str] = "RuntimeType.PYTORCH",
        team_id: int = None,
        timeout: int = 100,
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
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        :param team_id: Team ID where the artifacts are stored. If not provided, will be taken from the current context.
        :type team_id: Optional[int]
        :param timeout: Timeout in seconds (default is 100). The maximum time to wait for the serving app to be ready.
        :type timeout: Optional[int]
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :return: Task Info
        :rtype: Dict[str, Any]
        :raises ValueError: if validations fail.
        """
        from supervisely.nn.utils import ModelSource

        if not isinstance(artifacts_dir, str) or not artifacts_dir.strip():
            raise ValueError("artifacts_dir must be a non-empty string.")

        if team_id is None:
            team_id = env.team_id()
        logger.debug(
            f"Starting custom model deployment. Team: {team_id}, Artifacts Dir: '{artifacts_dir}'"
        )

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
            task_info = self._run_serve_app(agent_id, module_id, timeout=timeout, **kwargs)
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
        runtime: Optional[str] = "RuntimeType.PYTORCH",
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

        logger.debug(f"Starting model deployment from experiment info. Task ID: '{task_id}'")
        train_module_id = train_task_info["meta"]["app"]["moduleId"]
        module = self.get_serving_app_by_train_app(train_module_id)
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

    def _run_serve_app(self, agent_id: int, module_id, timeout: int = 100, **kwargs):
        _attempt_delay_sec = 1
        _attempts = timeout // _attempt_delay_sec

        task_info = self._api.task.start(agent_id=agent_id, module_id=module_id, **kwargs)
        ready = self._api.app.wait_until_ready_for_api_calls(
            task_info["id"], _attempts, _attempt_delay_sec
        )
        if not ready:
            raise TimeoutError(
                f"Task {task_info['id']} is not ready for API calls after {timeout} seconds."
            )
        return task_info

    def _load_model_from_api(self, task_id, deploy_params, model_name: Optional[str] = None):
        self._api.task.send_request(
            task_id,
            "deploy_from_api",
            data={"deploy_params": deploy_params, "model_name": model_name},
            raise_error=True,
        )

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
        from supervisely.nn.utils import ModelSource

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
        runtime: str,
        with_module: bool = True,
    ):
        from dataclasses import asdict

        from supervisely.nn.experiments import get_experiment_info_by_artifacts_dir
        from supervisely.nn.utils import ModelSource

        experiment_info = get_experiment_info_by_artifacts_dir(self._api, team_id, artifacts_dir)
        if not experiment_info:
            raise ValueError(
                f"Failed to retrieve experiment info for artifacts_dir: '{artifacts_dir}'"
            )

        experiment_task_id = experiment_info.task_id
        experiment_task_info = self._api.task.get_info_by_id(experiment_task_id)
        if experiment_task_info is None:
            raise ValueError(f"Task with ID '{experiment_task_id}' not found")

        module_id = None
        serve_app_name = None
        if with_module:
            train_module_id = experiment_task_info["meta"]["app"]["moduleId"]
            module = self.get_serving_app_by_train_app(train_module_id)
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
            logger.info("Checkpoint name not provided. Using the best checkpoint.")
            checkpoint = experiment_info.best_checkpoint

        model_info_dict = asdict(experiment_info)
        model_info_dict["artifacts_dir"] = artifacts_dir
        checkpoint_name = checkpoint
        deploy_params = {
            "device": device,
            "model_source": ModelSource.CUSTOM,
            "model_files": {
                "checkpoint": f"/{artifacts_dir.strip('/')}/checkpoints/{checkpoint_name}"
            },
            "model_info": model_info_dict,
            "runtime": runtime,
        }

        config = experiment_info.model_files.get("config")
        if config is not None:
            deploy_params["model_files"]["config"] = f"{experiment_info.artifacts_dir}{config}"
            logger.debug(f"Config file added: {experiment_info.artifacts_dir}{config}")
        return module_id, serve_app_name, deploy_params
