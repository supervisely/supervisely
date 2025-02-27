# coding: utf-8
"""download/upload/manipulate neural networks"""

from typing import TYPE_CHECKING, Optional, Union

from supervisely.api.neural_network.deploy_api import DeployApi

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

    # Deploy Models
    def deploy_pretrained_model(
        self,
        app_name: str,
        model_name: str,
        device: str = "cuda",
        runtime: str = "PyTorch",
        **kwargs,
    ) -> "Session":
        """
        Start a new serving app and deploy pretrained model by model_name.

        :param app_name: App name in Supervisely (e.g., "Serve RT-DETRv2").
        :type app_name: str
        :param model_name: Model name to deploy (e.g., "RT-DETRv2-M").
        :type model_name: str
        :param device: Device string (default is "cuda").
        :type device: str
        :param runtime: Runtime string (default is "PyTorch").
        :type runtime: str
        """
        from supervisely.nn.inference.session import Session

        task_info = self.deploy.deploy_pretrained_model(
            app_name=app_name,
            model_name=model_name,
            device=device,
            runtime=runtime,
            **kwargs,
        )
        return Session(self._api, task_info["id"])

    def deploy_custom_model(
        self,
        artifacts_dir: str,
        checkpoint_name: Optional[str] = None,
        device: str = "cuda",
        team_id: int = None,
        **kwargs,
    ) -> "Session":
        """
        Start a new serving app and deploy custom model based on the directory path in Team Files where the model is stored.

        :param workspace_id: Workspace ID in Supervisely.
        :type workspace_id: int
        :param artifacts_dir: Path to the artifacts directory in Team Files.
        :type artifacts_dir: str
        :param checkpoint_name: Checkpoint name (with file extension) to deploy, e.g. "best.pt".
            If not provided, checkpoint will be chosen automatically, trying to pick the "best" checkpoint if available.
        :type checkpoint_name: Optional[str]
        :param device: Device string (default is "cuda").
        :type device: str
        :param timeout: Timeout in seconds (default is 100).
        :type timeout: int
        :param kwargs: Additional parameters to start the task. See Api.task.start() for more details.
        :type kwargs: Dict[str, Any]
        :raises ValueError: if validations fail.
        :return: a :class:`Session` object
        :rtype: Session
        """
        from supervisely.nn.inference.session import Session

        task_info = self.deploy.deploy_custom_model_by_artifacts_dir(
            artifacts_dir=artifacts_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            team_id=team_id,
            **kwargs,
        )
        return Session(self._api, task_info["id"])

    def get_experiment_info(self, task_id: int) -> "ExperimentInfo":
        """
        Returns the experiment info of a finished training task by its task_id.

        :param task_id: the task_id of a finished training task in the Supervisely platform.
        :type task_id: int
        :return: a :class:`ExperimentInfo` object with information about the training, model, and results.
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

    def inference_session(
        self,
        session_id: int,
        inference_settings: Union[dict, str] = None,
    ) -> "Session":
        """
        Attach to a running Serving App session to run the inference via API.

        :param session_id: the session_id of a running Serving App session in the Supervisely platform.
        :type session_id: int
        :param inference_settings: a dict or a path to YAML file with settings, defaults to None
        :type inference_settings: Union[dict, str], optional
        :return: a :class:`Session` object
        :rtype: Session
        """
        from supervisely.nn.inference.session import Session

        return Session(self._api, session_id, inference_settings=inference_settings)
