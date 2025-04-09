# coding: utf-8
"""download/upload/manipulate neural networks"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from typing_extensions import Literal

from supervisely.api.neural_network.deploy_api import DeployApi
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo
    from supervisely.nn.model_api import ModelApi


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
        team_id: int = None,
        **kwargs,
    ) -> "ModelApi":
        """
        Deploy model by checkpoint path or model_name.

        :param model: Either path to the model checkpoint in team files or model name in format framework/model_name (e.g., "RT-DETRv2/RT-DETRv2-M").
        :type model: str
        :param device: Device string
        :type device: Optional[str]
        :param runtime: Runtime string, if not present will be defined automatically.
        :type runtime: Optional[str]
        """
        from supervisely.nn.model_api import ModelApi

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
                **kwargs,
            )
        else:
            framework, model_name = pretrained.split("/", 1)
            logger.debug(
                f"Deploying pretrained model: {model}. Framework: {framework}, Model name: {model_name}"
            )
            task_info = self._deploy_api.deploy_pretrained_model(
                framework=framework,
                model_name=model_name,
                device=device,
                runtime=runtime,
                team_id=team_id,
                **kwargs,
            )
        return ModelApi(self._api, deploy_id=task_info["id"])

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
        session_id: int,
        inference_settings: Union[dict, str] = None,
    ) -> "ModelApi":
        """
        Attach to a running Serving App session to run the inference via API.

        :param session_id: the session_id of a running Serving App session in the Supervisely platform.
        :type session_id: int
        :param inference_settings: a dict or a path to YAML file with settings, defaults to None
        :type inference_settings: Union[dict, str], optional
        :return: a :class:`Session` object
        :rtype: Session
        """
        from supervisely.nn.model_api import ModelApi

        return ModelApi(self._api, session_id)
