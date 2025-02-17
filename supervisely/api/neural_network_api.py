# coding: utf-8
"""download/upload/manipulate neural networks"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from supervisely.api.module_api import CloneableModuleApi, RemoveableModuleApi
from supervisely.io import env


class NeuralNetworkApi(CloneableModuleApi, RemoveableModuleApi):
    """ """

    def deploy_model_from_api(self, task_id, deploy_params):
        self._api.task.send_request(
            task_id, "deploy_from_api", data={"deploy_params": deploy_params}, raise_error=True
        )

    def deploy(
        self,
        module_id: Optional[int] = None,
        train_app_id: Optional[int] = None,
        train_module_id: Optional[int] = None,
        train_task_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        agent_id: Optional[int] = None,
        description: Optional[str] = "application description",
        params: Dict[str, Any] = None,
        log_level: Optional[Literal["info", "debug", "warning", "error"]] = "info",
        users_ids: Optional[List[int]] = None,
        app_version: Optional[str] = "",
        is_branch: Optional[bool] = False,
        task_name: Optional[str] = "pythonSpawned",
        restart_policy: Optional[Literal["never", "on_error"]] = "never",
        proxy_keep_url: Optional[bool] = False,
        redirect_requests: Optional[Dict[str, int]] = {},
        limit_by_workspace: bool = False,
        deploy_params: Dict[str, Any] = None,
        timeout: int = 100,
    ):
        if train_task_id is not None:
            if train_module_id or train_app_id:
                raise ValueError(
                    "train_module_id and train_app_id are not allowed with train_task_id"
                )
            task_info = self.get_info_by_id(train_task_id)
            try:
                data = task_info["meta"]["output"]["experiment"]["data"]
            except KeyError:
                raise ValueError("Task output does not contain experiment data")

            if module_id is None:
                train_module_id = task_info["meta"]["app"]["moduleId"]
                module_id = self._get_serving_app_module_id(train_module_id)

            if workspace_id is None:
                workspace_id = task_info["workspaceId"]

            experiment_name = data["experiment_name"]

            deploy_params_overwrite = deploy_params
            if deploy_params_overwrite is None:
                deploy_params_overwrite = {}

            if checkpoint_name is None:
                checkpoint_name = data["best_checkpoint"]

            if task_name is None:
                task_name = experiment_name + f" ({checkpoint_name})"

            if description is None:
                description = f"""Serve from experiment
                    Experiment name:   {experiment_name}
                    Evaluation report: {data["evaluation_report_link"]}
                """
                while len(description) > 255:
                    description = description.rsplit("\n", 1)[0]

            deploy_params = {
                "model_files": {
                    "checkpoint": Path(
                        data["artifacts_dir"], "checkpoints", checkpoint_name
                    ).as_posix(),
                    "config": Path(data["artifacts_dir"], data["model_files"]["config"]).as_posix(),
                },
                "model_source": "Custom models",
                "model_info": data,
                "device": "cuda",
                "runtime": "PyTorch",
            }
            deploy_params = {**deploy_params, **deploy_params_overwrite}
        elif train_module_id is not None:
            if module_id is not None:
                raise ValueError("module_id is not allowed with train_module_id")
            module_id = self._get_serving_app_module_id(train_module_id)
        elif train_app_id is not None:
            if module_id is not None:
                raise ValueError("module_id is not allowed with train_app_id")
            train_module_id = self._api.app.get_info_by_id(train_app_id).module_id
            module_id = self._get_serving_app_module_id(train_module_id)

        if workspace_id is None:
            workspace_id = env.workspace_id()
        return self._deploy_model_app(
            module_id=module_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            description=description,
            params=params,
            log_level=log_level,
            users_ids=users_ids,
            app_version=app_version,
            is_branch=is_branch,
            task_name=task_name,
            restart_policy=restart_policy,
            proxy_keep_url=proxy_keep_url,
            redirect_requests=redirect_requests,
            limit_by_workspace=limit_by_workspace,
            deploy_params=deploy_params,
            timeout=timeout,
        )

    def _deploy_model_app(
        self,
        module_id: int,
        workspace_id: int,
        agent_id: Optional[int] = None,
        description: Optional[str] = "application description",
        params: Dict[str, Any] = None,
        log_level: Optional[Literal["info", "debug", "warning", "error"]] = "info",
        users_ids: Optional[List[int]] = None,
        app_version: Optional[str] = "",
        is_branch: Optional[bool] = False,
        task_name: Optional[str] = "pythonSpawned",
        restart_policy: Optional[Literal["never", "on_error"]] = "never",
        proxy_keep_url: Optional[bool] = False,
        redirect_requests: Optional[Dict[str, int]] = {},
        limit_by_workspace: bool = False,
        deploy_params: Dict[str, Any] = None,
        timeout: int = 100,
    ):
        task_info = self._api.task.start(
            agent_id=agent_id,
            workspace_id=workspace_id,
            module_id=module_id,
            description=description,
            params=params,
            log_level=log_level,
            users_ids=users_ids,
            app_version=app_version,
            is_branch=is_branch,
            task_name=task_name,
            restart_policy=restart_policy,
            proxy_keep_url=proxy_keep_url,
            redirect_requests=redirect_requests,
            limit_by_workspace=limit_by_workspace,
        )

        attempt_delay_sec = 10
        attempts = (timeout + attempt_delay_sec) // attempt_delay_sec
        ready = self._api.app.wait_until_ready_for_api_calls(
            task_info["id"], attempts, attempt_delay_sec
        )
        if not ready:
            raise TimeoutError(
                f"Task {task_info['id']} is not ready for API calls after {timeout} seconds."
            )
        self.deploy_model_from_api(task_info["id"], deploy_params=deploy_params)
        return task_info

    def _get_serving_app_module_id(self, train_app_module_id: int):
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

        modules = self._api.app.get_list_ecosystem_modules(
            categories=["serve", framework], categories_operation="and"
        )
        if len(modules) == 0:
            raise ValueError(f"No serving apps found for framework {framework}")
        module_id = modules[0]["id"]
        return module_id
