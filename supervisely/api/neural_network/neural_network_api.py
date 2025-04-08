# coding: utf-8
"""download/upload/manipulate neural networks"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from supervisely.api.neural_network.deploy_api import DeployApi
from supervisely.api.neural_network.model_api import ModelApi

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo


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
        checkpoint: str = None,
        pretrained: str = None,
        app_name: str = None,
        device: str = None,
        runtime: Optional[str] = "PyTorch",
        team_id: int = None,
        **kwargs,
    ) -> ModelApi:
        """
        Deploy model by checkpoint path or model_name.

        :param app_name: App name in Supervisely (e.g., "Serve RT-DETRv2").
        :type app_name: str
        :param model_name: Model name to deploy (e.g., "RT-DETRv2-M").
        :type model_name: str
        :param device: Device string (default is "cuda").
        :type device: str
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        """
        assert (
            checkpoint is not None or pretrained is not None
        ), "Either checkpoint or pretrained model name must be provided."
        assert (
            checkpoint is None or pretrained is None
        ), "Only one of checkpoint or pretrained model name can be provided."
        assert (
            pretrained is None or app_name is not None
        ), "App name must be provided for pretrained models."

        if checkpoint is not None:
            task_info = self._deploy_api.deploy_custom_model_by_checkpoint(
                checkpoint=checkpoint,
                device=device,
                runtime=runtime,
                team_id=team_id,
                **kwargs,
            )
        else:
            task_info = self._deploy_api.deploy_pretrained_model(
                app_name=app_name,
                model_name=pretrained,
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
    ) -> ModelApi:
        """
        Attach to a running Serving App session to run the inference via API.

        :param session_id: the session_id of a running Serving App session in the Supervisely platform.
        :type session_id: int
        :param inference_settings: a dict or a path to YAML file with settings, defaults to None
        :type inference_settings: Union[dict, str], optional
        :return: a :class:`Session` object
        :rtype: Session
        """
        return ModelApi(self._api, session_id, params=inference_settings)
