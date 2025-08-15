# coding: utf-8
"""deploy and connect to running serving apps"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import supervisely.io.env as sly_env
import supervisely.io.env as env
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo
    from supervisely.nn.model.model_api import ModelAPI


class NeuralNetworkApi:
    """
    API to interact with neural networks in Supervisely.
    It provides methods to deploy and connect to models for running inference.
    """

    def __init__(self, api: "Api"):
        from supervisely.api.nn.deploy_api import DeployApi
        from supervisely.api.nn.ecosystem_models_api import EcosystemModelsApi

        self._api = api
        self._deploy_api = DeployApi(api)
        self.ecosystem_models_api = EcosystemModelsApi(api)

    def deploy(
        self,
        model: str,
        device: str = None,
        runtime: str = None,
        workspace_id: int = None,
        agent_id: int = None,
        **kwargs,
    ) -> ModelAPI:
        """
        Deploy a pretrained model or a custom model checkpoint in Supervisely platform.
        This method will start a new Serving App in Supervisely, deploy a given model, and return a `ModelAPI` object for running predictions and managing the model.
        - To deploy a pretrained model, pass the model name in the format `framework/model_name` (e.g., "RT-DETRv2/RT-DETRv2-M").
        - To deploy a custom model, pass the path to the model checkpoint in team files (e.g., "/experiments/1089_RT-DETRv2/checkpoints/best.pt").

        :param model: Either a path to a model checkpoint in team files or model name in format `framework/model_name` (e.g., "RT-DETRv2/RT-DETRv2-M").
        :type model: str
        :param device: Device to run the model on (e.g., "cuda:0" or "cpu"). If not specified, will automatically use GPU device if available, otherwise CPU will be used.
        :type device: Optional[str]
        :param runtime: If specified, the model will be converted to the given format (e.g., "onnx", "tensorrt") and will be deployed in the corresponding accelerated runtime. This option is used for pretrained models. For custom models, the runtime will be defined automatically based on the model checkpoint.
        :type runtime: Optional[str]
        :param workspace_id: Workspace ID, if None, will be got from env.
        :type workspace_id: Optional[int]
        :param agent_id: Agent ID, if not present will be defined automatically.
        :type agent_id: Optional[int]
        :param kwargs: Additional parameters for deployment.
        :return: A :class:`ModelAPI` object for the deployed model.
        :rtype: ModelAPI
        :Usage example:
            .. code-block:: python

                import supervisely as sly

                api = sly.Api()
                model = api.nn.deploy(model="RT-DETRv2/RT-DETRv2-M")
        """

        from supervisely.nn.model.model_api import ModelAPI

        checkpoint = None
        pretrained = None
        team_id = None
        if workspace_id is None:
            workspace_id = sly_env.workspace_id(raise_not_found=False)
        if workspace_id is None:
            raise ValueError(
                "Workspace ID is not specified. Please, provide it in the function call, or set 'WORKSPACE_ID' variable in the environment."
            )
        if team_id is None:
            workspace_info = self._api.workspace.get_info_by_id(workspace_id)
            team_id = workspace_info.team_id
        if agent_id is None:
            agent_id = self._deploy_api._find_agent(team_id)

        if model.startswith("/"):
            checkpoint = model
        else:
            found_team_id = self._deploy_api._find_team_by_path(
                f"/{model}", team_id=team_id, raise_not_found=False
            )
            if found_team_id is not None:
                checkpoint = f"/{model}"
                team_id = found_team_id
                logger.debug(f"Found checkpoint in team {team_id}")
            else:
                pretrained = model

        if checkpoint is not None:
            logger.debug(f"Deploying model by checkpoint: {checkpoint}")
            task_info = self._deploy_api.deploy_custom_model_by_checkpoint(
                checkpoint=checkpoint,
                device=device,
                runtime=runtime,
                team_id=team_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                **kwargs,
            )
        else:
            framework, model_name = pretrained.split("/", 1)
            logger.debug(
                f"Deploying pretrained model. Framework: {framework}, Model name: {model_name}"
            )
            task_info = self._deploy_api.deploy_pretrained_model(
                framework=framework,
                model_name=model_name,
                device=device,
                runtime=runtime,
                workspace_id=workspace_id,
                agent_id=agent_id,
                **kwargs,
            )
        return ModelAPI(self._api, task_id=task_info["id"])

    def list_deployed_models(
        self,
        model: str = None,
        framework: str = None,
        task_type: str = None,
        team_id: int = None,
        workspace_id: int = None,
    ) -> List[Dict]:
        """
        Returns a list of deployed models in the Supervisely platform.
        The list can be filtered by model name, framework, task type, team ID, and workspace ID.

        :param model: Model name or checkpoint path to filter the results. If None, all models will be returned.
        :type model: Optional[str]
        :param framework: Framework name to filter the results. If None, all frameworks will be returned.
        :type framework: Optional[str]
        :param task_type: CV Task to filter the results, e.g., "object detection", "instance segmentation", etc. If None, all task types will be returned.
        :type task_type: Optional[str]
        :param team_id: Team ID to filter the results. If None, the team ID from the environment will be used.
        :type team_id: Optional[int]
        :param workspace_id: Workspace ID to filter the results. If None, the workspace ID from the environment will be used.
        :type workspace_id: Optional[int]
        :return: A list of dictionaries containing information about the deployed models.
        :rtype: List[Dict]
        :Usage example:
            .. code-block:: python

                import supervisely as sly

                api = sly.Api()
                deployed_models = api.nn.list_deployed_models(framework="RT-DETRv2")
        """
        # 1. Define apps
        categories = ["serve"]
        if framework is not None:
            categories.append(f"framework:{framework}")
        serve_apps = self._api.app.get_list_ecosystem_modules(
            categories=categories, categories_operation="and"
        )
        if not serve_apps:
            return []
        serve_apps_module_ids = {app["id"] for app in serve_apps}
        # 2. Get tasks infos
        if workspace_id is not None:
            workspaces = [workspace_id]
        elif team_id is not None:
            workspaces = self._api.workspace.get_list(team_id)
            workspaces = [workspace.id for workspace in workspaces]
        else:
            workspace_id = env.workspace_id(raise_not_found=False)
            if workspace_id is None:
                team_id = env.team_id(raise_not_found=False)
                if team_id is None:
                    raise ValueError(
                        "Workspace ID and Team ID are not specified and cannot be found in the environment."
                    )
                workspaces = self._api.workspace.get_list(team_id)
                workspaces = [workspace.id for workspace in workspaces]
            else:
                workspaces = [workspace_id]

        all_tasks = []
        for workspace_id in workspaces:
            all_tasks.extend(
                self._api.task.get_list(
                    workspace_id=workspace_id,
                    filters=[
                        {
                            "field": "status",
                            "operator": "in",
                            "value": [str(self._api.task.Status.STARTED)],
                        }
                    ],
                )
            )
        all_tasks = [
            task for task in all_tasks if task["meta"]["app"]["moduleId"] in serve_apps_module_ids
        ]
        # get deploy infos and filter results
        result = []
        for task in all_tasks:
            try:
                deploy_info = self._deploy_api.get_deploy_info(task["id"])
            except Exception as e:
                logger.warning(
                    f"Failed to get deploy info for task {task['id']}: {e}",
                    exc_info=True,
                )
                continue
            if model is not None:
                checkpoint = deploy_info["checkpoint_name"]
                deployed_model = deploy_info["model_name"]
                if checkpoint != model and not model.endswith(deployed_model):
                    continue
            if task_type is not None:
                if deploy_info["task_type"] != task_type:
                    continue
            result.append(
                {
                    "task_info": task,
                    "model_info": deploy_info,
                }
            )
        return result

    def get_experiment_info(self, task_id: int) -> ExperimentInfo:
        """
        Returns the experiment info of a finished training task by its task_id.

        :param task_id: the task_id of a finished training task in the Supervisely platform.
        :type task_id: int
        :return: an :class:`ExperimentInfo` object with information about the training, model, and results.
        :rtype: ExperimentInfo
        """
        from supervisely.nn.experiments import ExperimentInfo

        task_info = self._api.task.get_info_by_id(task_id)
        if task_info is None:
            raise ValueError(f"Task with ID '{task_id}' not found")
        try:
            data = task_info["meta"]["output"]["experiment"]["data"]
            return ExperimentInfo(**data)
        except KeyError:
            raise ValueError("Task output does not contain experiment data")

    def connect(
        self,
        task_id: int,
    ) -> ModelAPI:
        """
        Connect to a running Serving App by its `task_id`. This allows you to make predictions and control the model state via API.

        :param task_id: the task_id of a running Serving App session in the Supervisely platform.
        :type task_id: int
        :return: a :class:`ModelAPI` object
        :rtype: ModelAPI
        """
        from supervisely.nn.model.model_api import ModelAPI

        return ModelAPI(self._api, task_id)
