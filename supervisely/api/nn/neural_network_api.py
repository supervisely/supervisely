# coding: utf-8
"""download/upload/manipulate neural networks"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from typing_extensions import Literal

from supervisely.api.nn.deploy_api import DeployApi
from supervisely.sly_logger import logger
import supervisely.io.env as sly_env
if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo
    from supervisely.nn.model.model_api import ModelAPI


class NeuralNetworkApi:
    """
    API to interact with neural networks in Supervisely.
    It provides methods to deploy models and run inference.
    """

    def __init__(self, api: "Api"):
        self._api = api
        self._deploy_api = DeployApi(api)

    def deploy(
        self,
        model: str,
        device: str = None,
        runtime: str = None,
        workspace_id: int = None,
        agent_id: int = None,
        **kwargs,
    ) -> "ModelAPI":
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
        :param workspace_id: Workspace ID, if not present will be defined automatically.
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
            workspace_id = sly_env.workspace_id()
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

    def get_deployed_models(
        self,
        workspace_id: int,
        model_name: str = None,
        framework: str = None,
        model_id: str = None,
        checkpoint: str = None,
        model: str = None,
        task_type: str = None,
    ) -> List[Dict]:
        if not any([model_name, framework, model_id, checkpoint, model, task_type]):
            raise ValueError("At least one filter parameter must be provided")
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
        all_tasks = self._api.task.get_list(
            workspace_id=workspace_id,
            filters=[
                {"field": "status", "operator": "in", "value": [str(self._api.task.Status.STARTED)]}
            ],
        )
        all_tasks = [
            task for task in all_tasks if task["meta"]["app"]["moduleId"] in serve_apps_module_ids
        ]
        # get deploy infos and filter results
        result = []
        for task in all_tasks:
            deploy_info = self._deploy_api.get_deploy_info(task["id"])
            if model_name is not None:
                if deploy_info["model_name"] != model_name:
                    continue
            if checkpoint is not None:
                if deploy_info["checkpoint_name"] != checkpoint:
                    continue
            if task_type is not None:
                if deploy_info["task_type"] != task_type:
                    continue
            result.append(
                {
                    "task_info": task,
                    "deploy_info": deploy_info,
                }
            )
        return result

    def get_experiment_info(self, task_id: int) -> "ExperimentInfo":
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
    ) -> "ModelAPI":
        """
        Connect to a running Serving App by `task_id`. This allows you to make predictions and control the model state via API.

        :param task_id: the task_id of a running Serving App session in the Supervisely platform.
        :type task_id: int
        :return: a :class:`ModelAPI` object
        :rtype: ModelAPI
        """
        from supervisely.nn.model.model_api import ModelAPI

        return ModelAPI(self._api, task_id)
