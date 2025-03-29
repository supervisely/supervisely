# coding: utf-8
"""download/upload/manipulate neural networks"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from supervisely.api.neural_network.deploy_api import DeployApi
from supervisely.api.neural_network.model_api import ModelApi

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo
    from supervisely.nn.inference.session import Session


class NeuralNetworkApi:
    """
    API to interact with neural networks in Supervisely.
    It provides methods to deploy models and run inference.
    """

    def __init__(self, api: "Api"):
        self._api = api
        self.deploy = DeployApi(api)

    def deploy_pretrained_model(
        self,
        agent_id: int,
        app_name: str,
        model_name: str,
        device: Optional[str] = None,
        runtime: Optional[str] = "PyTorch",
        **kwargs,
    ) -> "Session":
        """
        Deploy pretrained model by model name.

        :param app_name: App name in Supervisely (e.g., "Serve RT-DETRv2").
        :type app_name: str
        :param model_name: Model name to deploy (e.g., "RT-DETRv2-M").
        :type model_name: str
        :param device: Device string (default is "cuda").
        :type device: str
        :param runtime: Runtime string, default is "PyTorch".
        :type runtime: Optional[str]
        """
        from supervisely.nn.inference.session import Session

        task_info = self.deploy.deploy_pretrained_model(
            agent_id=agent_id,
            app=app_name,
            model_name=model_name,
            device=device,
            runtime=runtime,
            **kwargs,
        )
        return Session(self._api, task_info["id"])

    def deploy_custom_model(
        self,
        agent_id: int,
        artifacts_dir: str,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        team_id: int = None,
        **kwargs,
    ) -> "Session":
        """
        Deploy custom model based on the directory path in Team Files where the artifacts are stored.

        :param artifacts_dir: Path to the artifacts directory in Team Files.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, checkpoint will be chosen automatically, trying to pick the "best" checkpoint if available.
        :type checkpoint_name: Optional[str]
        :param device: Device string. If not provided, will be chosen automatically.
        :type device: Optional[str]
        :param team_id: Team ID where the artifacts are stored. If not provided, will be taken from the current context.
        :type team_id: int
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :raises ValueError: if validations fail.
        :return: a :class:`Session` object
        :rtype: Session
        """
        from supervisely.nn.inference.session import Session

        task_info = self.deploy.deploy_custom_model_by_artifacts_dir(
            agent_id=agent_id,
            artifacts_dir=artifacts_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            team_id=team_id,
            **kwargs,
        )
        return Session(self._api, task_info["id"])

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
            deploy_info = self.deploy.get_deploy_info(task["id"])
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
