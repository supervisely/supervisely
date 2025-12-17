# coding: utf-8
"""api for working with tasks"""

import json
import os
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

# docs
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union

import requests
from pydantic import BaseModel, Field
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from supervisely import logger
from supervisely._utils import batched, take_with_default
from supervisely.api.module_api import (
    ApiField,
    ModuleApiBase,
    ModuleWithStatus,
    WaitingTimeExceeded,
)
from supervisely.collection.str_enum import StrEnum
from supervisely.io.env import app_categories
from supervisely.io.fs import (
    ensure_base_path,
    get_file_hash,
    get_file_name,
    get_file_name_with_ext,
)


class KubernetesSettings(BaseModel):
    """
    KubernetesSettings for application resource limits and requests.
    """

    use_health_check: Optional[bool] = Field(None, alias="useHealthCheck")
    request_cpus: Optional[int] = Field(None, alias="requestCpus")
    limit_cpus: Optional[int] = Field(None, alias="limitCpus")
    limit_memory_gb: Optional[int] = Field(None, alias="limitMemoryGb")
    limit_shm_gb: Optional[int] = Field(None, alias="limitShmGb")
    limit_storage_gb: Optional[int] = Field(None, alias="limitStorageGb")
    limit_gpus: Optional[int] = Field(None, alias="limitGpus")
    limit_gpu_memory_mb: Optional[int] = Field(None, alias="limitGpuMemoryMb")
    limit_gpu_cores_perc: Optional[int] = Field(None, alias="limitGpuCoresPerc")

    model_config = {"populate_by_name": True}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with only non-None values using aliases."""
        return self.model_dump(exclude_none=True, by_alias=True)


class TaskFinishedWithError(Exception):
    """TaskFinishedWithError"""

    pass


class TaskApi(ModuleApiBase, ModuleWithStatus):
    """
    API for working with Tasks. :class:`TaskApi<TaskApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        task_id = 121230
        task_info = api.task.get_info_by_id(task_id)
    """

    class RestartPolicy(StrEnum):
        """RestartPolicy"""

        NEVER = "never"
        """"""
        ON_ERROR = "on_error"
        """"""

    class PluginTaskType(StrEnum):
        """PluginTaskType"""

        TRAIN = "train"
        """"""
        INFERENCE = "inference"
        """"""
        INFERENCE_RPC = "inference_rpc"
        """"""
        SMART_TOOL = "smarttool"
        """"""
        CUSTOM = "custom"
        """"""

    class Status(StrEnum):
        """Status"""

        QUEUED = "queued"
        """Application is queued for execution"""
        CONSUMED = "consumed"
        """Application is consumed by an agent"""
        STARTED = "started"
        """Application has been started"""
        DEPLOYED = "deployed"
        """Only for Plugins"""
        ERROR = "error"
        """Application has finished with an error"""
        FINISHED = "finished"
        """Application has finished successfully"""
        TERMINATING = "terminating"
        """Application is being terminated"""
        STOPPED = "stopped"
        """Application has been stopped"""

    def __init__(self, api):
        ModuleApiBase.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(
        self, workspace_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[NamedTuple]:
        """
        List of Tasks in the given Workspace.

        :param workspace_id: Workspace ID.
        :type workspace_id: int
        :param filters: List of params to sort output Projects.
        :type filters: List[dict], optional
        :return: List of Tasks with information for the given Workspace.
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            workspace_id = 23821

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_infos = api.task.get_list(workspace_id)

            task_infos_filter = api.task.get_list(23821, filters=[{'field': 'id', 'operator': '=', 'value': 121230}])
            print(task_infos_filter)
            # Output: [
            #     {
            #         "id": 121230,
            #         "type": "clone",
            #         "status": "finished",
            #         "startedAt": "2019-12-19T12:13:09.702Z",
            #         "finishedAt": "2019-12-19T12:13:09.701Z",
            #         "meta": {
            #             "input": {
            #                 "model": {
            #                     "id": 1849
            #                 },
            #                 "isExternal": true,
            #                 "pluginVersionId": 84479
            #             },
            #             "output": {
            #                 "model": {
            #                     "id": 12380
            #                 },
            #                 "pluginVersionId": 84479
            #             }
            #         },
            #         "description": ""
            #     }
            # ]
        """
        return self.get_list_all_pages(
            "tasks.list",
            {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []},
        )

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Task information by ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Information about Task.
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_info = api.task.get_info_by_id(task_id)
            print(task_info)
            # Output: {
            #     "id": 121230,
            #     "workspaceId": 23821,
            #     "description": "",
            #     "type": "clone",
            #     "status": "finished",
            #     "startedAt": "2019-12-19T12:13:09.702Z",
            #     "finishedAt": "2019-12-19T12:13:09.701Z",
            #     "userId": 16154,
            #     "meta": {
            #         "app": {
            #             "id": 10370,
            #             "name": "Auto Import",
            #             "version": "test-branch",
            #             "isBranch": true,
            #         },
            #         "input": {
            #             "model": {
            #                 "id": 1849
            #             },
            #             "isExternal": true,
            #             "pluginVersionId": 84479
            #         },
            #         "output": {
            #             "model": {
            #                 "id": 12380
            #             },
            #             "pluginVersionId": 84479
            #         }
            #     },
            #     "settings": {},
            #     "agentName": null,
            #     "userLogin": "alexxx",
            #     "teamId": 16087,
            #     "agentId": null
            # }
        """
        return self._get_info_by_id(id, "tasks.info")

    def get_status(self, task_id: int) -> Status:
        """
        Check status of Task by ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Status object
        :rtype: :class:`Status`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            task_status = api.task.get_status(task_id)
            print(task_status)
            # Output: finished
        """
        status_str = self.get_info_by_id(task_id)[ApiField.STATUS]  # @TODO: convert json to tuple
        return self.Status(status_str)

    def raise_for_status(self, status: Status) -> None:
        """
        Raise error if Task status is ERROR.

        :param status: Status object.
        :type status: Status
        :return: None
        :rtype: :class:`NoneType`
        """
        if status is self.Status.ERROR:
            raise TaskFinishedWithError(f"Task finished with status {str(self.Status.ERROR)}")

    def wait(
        self,
        id: int,
        target_status: Status,
        wait_attempts: Optional[int] = None,
        wait_attempt_timeout_sec: Optional[int] = None,
    ):
        """
        Awaiting achievement by given Task of a given status.

        :param id: Task ID in Supervisely.
        :type id: int
        :param target_status: Status object(status of task we expect to destinate).
        :type target_status: Status
        :param wait_attempts: The number of attempts to determine the status of the task that we are waiting for.
        :type wait_attempts: int, optional
        :param wait_attempt_timeout_sec: Number of seconds for intervals between attempts(raise error if waiting time exceeded).
        :type wait_attempt_timeout_sec: int, optional
        :return: True if the desired status is reached, False otherwise
        :rtype: :class:`bool`
        """
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status in [
                target_status,
                self.Status.FINISHED,
                self.Status.DEPLOYED,
                self.Status.STOPPED,
            ]:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded(
            f"Waiting time exceeded: total waiting time {wait_attempts * effective_wait_timeout} seconds, i.e. {wait_attempts} attempts for {effective_wait_timeout} seconds each"
        )

    def get_context(self, id: int) -> Dict:
        """
        Get context information by task ID.

        :param id: Task ID in Supervisely.
        :type id: int
        :return: Context information in dict format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            task_id = 121230

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            context = api.task.get_context(task_id)
            print(context)
            # Output: {
            #     "team": {
            #         "id": 16087,
            #         "name": "alexxx"
            #     },
            #     "workspace": {
            #         "id": 23821,
            #         "name": "my_super_workspace"
            #     }
            # }
        """
        response = self._api.post("GetTaskContext", {ApiField.ID: id})
        return response.json()

    def _convert_json_info(self, info: dict):
        """_convert_json_info"""
        return info

    def start(
        self,
        agent_id,
        app_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        description: Optional[str] = "application description",
        params: Dict[str, Any] = None,
        log_level: Optional[Literal["info", "debug", "warning", "error"]] = "info",
        users_ids: Optional[List[int]] = None,
        app_version: Optional[str] = "",
        is_branch: Optional[bool] = False,
        task_name: Optional[str] = "pythonSpawned",
        restart_policy: Optional[Literal["never", "on_error"]] = "never",
        proxy_keep_url: Optional[bool] = False,
        module_id: Optional[int] = None,
        redirect_requests: Optional[Dict[str, int]] = {},
        limit_by_workspace: bool = False,
        kubernetes_settings: Optional[Union[KubernetesSettings, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Starts the application task on the agent.

        :param agent_id: Agent ID. Can be obtained from TeamCluster page in UI.
        :type agent_id: int
        :param app_id: Deprecated. Use module_id instead.
        :type app_id: int, optional
        :param workspace_id: Workspace ID where the task will be created.
        :type workspace_id: int, optional
        :param description: Task description which will be shown in UI.
        :type description: str, optional
        :param params: Task parameters which will be passed to the application, check the
            code example below for more details.
        :type params: Dict[str, Any], optional
        :param log_level: Log level for the application.
        :type log_level: Literal["info", "debug", "warning", "error"], optional
        :param users_ids: List of user IDs for which will be created an instance of the application.
            For each user a separate task will be created.
        :type users_ids: List[int], optional
        :param app_version: Application version e.g. "v1.0.0" or branch name e.g. "dev".
        :type app_version: str, optional
        :param is_branch: If the application version is a branch name, set this parameter to True.
        :type is_branch: bool, optional
        :param task_name: Task name which will be shown in UI.
        :type task_name: str, optional
        :param restart_policy: when the task should be restarted: never or if error occurred.
        :type restart_policy: Literal["never", "on_error"], optional
        :param proxy_keep_url: For internal usage only.
        :type proxy_keep_url: bool, optional
        :param module_id: Module ID. Can be obtained from the apps page in UI.
        :type module_id: int, optional
        :param redirect_requests: For internal usage only in Develop and Debug mode.
        :type redirect_requests: Dict[str, int], optional
        :param limit_by_workspace: If set to True tasks will be only visible inside of the workspace
            with specified workspace_id.
        :type limit_by_workspace: bool, optional
        :param kubernetes_settings: Kubernetes settings for the application.
        :type kubernetes_settings: Union[KubernetesSettings, Dict[str, Any]], optional
        :return: Task information in JSON format.
        :rtype: Dict[str, Any]

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            app_slug = "supervisely-ecosystem/export-to-supervisely-format"
            module_id = api.app.get_ecosystem_module_id(app_slug)
            module_info = api.app.get_ecosystem_module_info(module_id)

            project_id = 12345
            agent_id = 12345
            workspace_id = 12345

            params = module_info.get_arguments(images_project=project_id)

            session = api.app.start(
                agent_id=agent_id,
                module_id=module_id,
                workspace_id=workspace_id,
                task_name="Prepare download link",
                params=params,
                app_version="dninja",
                is_branch=True,
            )
        """
        if app_id is not None and module_id is not None:
            raise ValueError("Only one of the arguments (app_id or module_id) have to be defined")
        if app_id is None and module_id is None:
            raise ValueError("One of the arguments (app_id or module_id) have to be defined")

        advanced_settings = {
            ApiField.LIMIT_BY_WORKSPACE: limit_by_workspace,
        }

        if kubernetes_settings is not None:
            if isinstance(kubernetes_settings, KubernetesSettings):
                kubernetes_settings = kubernetes_settings.to_dict()
            if not isinstance(kubernetes_settings, dict):
                raise TypeError(
                    f"kubernetes_settings must be a dict or an instance of KubernetesSettings, got {type(kubernetes_settings)}"
                )
            advanced_settings.update(kubernetes_settings)

        data = {
            ApiField.AGENT_ID: agent_id,
            # "nodeId": agent_id,
            ApiField.WORKSPACE_ID: workspace_id,
            ApiField.DESCRIPTION: description,
            ApiField.PARAMS: take_with_default(params, {"state": {}}),
            ApiField.LOG_LEVEL: log_level,
            ApiField.USERS_IDS: take_with_default(users_ids, []),
            ApiField.APP_VERSION: app_version,
            ApiField.IS_BRANCH: is_branch,
            ApiField.TASK_NAME: task_name,
            ApiField.RESTART_POLICY: restart_policy,
            ApiField.PROXY_KEEP_URL: proxy_keep_url,
            ApiField.ADVANCED_SETTINGS: advanced_settings,
        }
        if len(redirect_requests) > 0:
            data[ApiField.REDIRECT_REQUESTS] = redirect_requests

        if app_id is not None:
            data[ApiField.APP_ID] = app_id
        if module_id is not None:
            data[ApiField.MODULE_ID] = module_id
        resp = self._api.post(method="tasks.run.app", data=data)
        task = resp.json()[0]
        if "id" not in task:
            task["id"] = task.get("taskId")
        return task

    def stop(self, id: int):
        """stop"""
        response = self._api.post("tasks.stop", {ApiField.ID: id})
        return self.Status(response.json()[ApiField.STATUS])

    def submit_logs(self, logs) -> None:
        """submit_logs"""
        response = self._api.post("tasks.logs.add", {ApiField.LOGS: logs})
        # return response.json()[ApiField.TASK_ID]

    def upload_files(
        self,
        task_id: int,
        abs_paths: List[str],
        names: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """upload_files"""
        if len(abs_paths) != len(names):
            raise RuntimeError("Inconsistency: len(abs_paths) != len(names)")

        hashes = []
        if len(abs_paths) == 0:
            return

        hash_to_items = defaultdict(list)
        hash_to_name = defaultdict(list)
        for idx, item in enumerate(zip(abs_paths, names)):
            path, name = item
            item_hash = get_file_hash(path)
            hashes.append(item_hash)
            hash_to_items[item_hash].append(path)
            hash_to_name[item_hash].append(name)

        unique_hashes = set(hashes)
        remote_hashes = self._api.image.check_existing_hashes(list(unique_hashes))
        new_hashes = unique_hashes - set(remote_hashes)

        # @TODO: upload remote hashes
        if len(remote_hashes) != 0:
            files = []
            for hash in remote_hashes:
                for name in hash_to_name[hash]:
                    files.append({ApiField.NAME: name, ApiField.HASH: hash})
            for batch in batched(files):
                resp = self._api.post(
                    "tasks.files.bulk.add-by-hash",
                    {ApiField.TASK_ID: task_id, ApiField.FILES: batch},
                )
        if progress_cb is not None:
            progress_cb(len(remote_hashes))

        for batch in batched(list(zip(abs_paths, names, hashes))):
            content_dict = OrderedDict()
            for idx, item in enumerate(batch):
                path, name, hash = item
                if hash in remote_hashes:
                    continue
                content_dict["{}".format(idx)] = json.dumps({"fullpath": name, "hash": hash})
                content_dict["{}-file".format(idx)] = (name, open(path, "rb"), "")

            if len(content_dict) > 0:
                encoder = MultipartEncoder(fields=content_dict)
                resp = self._api.post("tasks.files.bulk.upload", encoder)
                if progress_cb is not None:
                    progress_cb(len(content_dict))

    # {
    #     data: {my_val: 1}
    #     obj: {val: 1, res: 2}
    # }
    # {
    #     obj: {new_val: 1}
    # }
    # // apped: true, recursive: false
    # {
    #     data: {my_val: 1}
    #     obj: {new_val: 1}
    # }(edited)
    # // append: false, recursive: false
    # {
    #     obj: {new_val: 1}
    # }(edited)
    #
    # 16: 32
    # // append: true, recursive: true
    # {
    #     data: {my_val: 1}
    #     obj: {val: 1, res: 2, new_val: 1}
    # }

    def set_fields(self, task_id: int, fields: List) -> Dict:
        """set_fields"""
        for idx, obj in enumerate(fields):
            for key in [ApiField.FIELD, ApiField.PAYLOAD]:
                if key not in obj:
                    raise KeyError("Object #{} does not have field {!r}".format(idx, key))
        data = {ApiField.TASK_ID: task_id, ApiField.FIELDS: fields}
        resp = self._api.post("tasks.data.set", data)
        return resp.json()

    def set_fields_from_dict(self, task_id: int, d: Dict) -> Dict:
        """set_fields_from_dict"""
        fields = []
        for k, v in d.items():
            fields.append({ApiField.FIELD: k, ApiField.PAYLOAD: v})
        return self.set_fields(task_id, fields)

    def set_field(
        self,
        task_id: int,
        field: Dict,
        payload: Dict,
        append: Optional[bool] = False,
        recursive: Optional[bool] = False,
    ) -> Dict:
        """set_field"""
        fields = [
            {
                ApiField.FIELD: field,
                ApiField.PAYLOAD: payload,
                ApiField.APPEND: append,
                ApiField.RECURSIVE: recursive,
            }
        ]
        return self.set_fields(task_id, fields)

    def get_fields(self, task_id, fields: List):
        """get_fields"""
        data = {ApiField.TASK_ID: task_id, ApiField.FIELDS: fields}
        resp = self._api.post("tasks.data.get", data)
        return resp.json()["result"]

    def get_field(self, task_id: int, field: str):
        """get_field"""
        result = self.get_fields(task_id, [field])
        return result[field]

    def _set_output(self):
        """_set_output"""
        pass

    def set_output_project(
        self,
        task_id: int,
        project_id: int,
        project_name: Optional[str] = None,
        project_preview: Optional[str] = None,
    ) -> Dict:
        """set_output_project"""
        if "import" in app_categories():
            self._api.project.add_import_history(project_id, task_id)
        if project_name is None:
            project = self._api.project.get_info_by_id(project_id, raise_error=True)
            project_name = project.name
            project_preview = project.image_preview_url

        output = {ApiField.PROJECT: {ApiField.ID: project_id, ApiField.TITLE: project_name}}
        if project_preview is not None:
            output[ApiField.PROJECT][ApiField.PREVIEW] = project_preview
        resp = self._api.post(
            "tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output}
        )
        return resp.json()

    def set_output_report(
        self,
        task_id: int,
        file_id: int,
        file_name: str,
        description: Optional[str] = "Report",
    ) -> Dict:
        """set_output_report"""
        return self._set_custom_output(
            task_id,
            file_id,
            file_name,
            description=description,
            icon="zmdi zmdi-receipt",
        )

    def _set_custom_output(
        self,
        task_id,
        file_id,
        file_name,
        file_url=None,
        description="File",
        icon="zmdi zmdi-file-text",
        color="#33c94c",
        background_color="#d9f7e4",
        download=False,
    ):
        """_set_custom_output"""
        if file_url is None:
            file_url = self._api.file.get_url(file_id)

        output = {
            ApiField.GENERAL: {
                "icon": {
                    "className": icon,
                    "color": color,
                    "backgroundColor": background_color,
                },
                "title": file_name,
                "titleUrl": file_url,
                "download": download,
                "description": description,
            }
        }
        resp = self._api.post(
            "tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output}
        )
        return resp.json()

    def set_output_archive(
        self, task_id: int, file_id: int, file_name: str, file_url: Optional[str] = None
    ) -> Dict:
        """set_output_archive"""
        if file_url is None:
            file_url = self._api.file.get_info_by_id(file_id).storage_path
        return self._set_custom_output(
            task_id,
            file_id,
            file_name,
            file_url=file_url,
            description="Download archive",
            icon="zmdi zmdi-archive",
            download=True,
        )

    def set_output_file_download(
        self,
        task_id: int,
        file_id: int,
        file_name: str,
        file_url: Optional[str] = None,
        download: Optional[bool] = True,
    ) -> Dict:
        """set_output_file_download"""
        if file_url is None:
            file_url = self._api.file.get_info_by_id(file_id).storage_path
        return self._set_custom_output(
            task_id,
            file_id,
            file_name,
            file_url=file_url,
            description="Download file",
            icon="zmdi zmdi-file",
            download=download,
        )

    def send_request(
        self,
        task_id: int,
        method: str,
        data: Dict,
        context: Optional[Dict] = {},
        skip_response: bool = False,
        timeout: Optional[int] = 60,
        outside_request: bool = True,
        retries: int = 10,
        raise_error: bool = False,
    ):
        """send_request"""
        if type(data) is not dict:
            raise TypeError("data argument has to be a dict")
        context["outside_request"] = outside_request
        resp = self._api.post(
            "tasks.request.direct",
            {
                ApiField.TASK_ID: task_id,
                ApiField.COMMAND: method,
                ApiField.CONTEXT: context,
                ApiField.STATE: data,
                "skipResponse": skip_response,
                "timeout": timeout,
            },
            retries=retries,
            raise_error=raise_error,
        )
        return resp.json()

    def set_output_directory(self, task_id, file_id, directory_path):
        """set_output_directory"""
        return self._set_custom_output(
            task_id,
            file_id,
            directory_path,
            description="Directory",
            icon="zmdi zmdi-folder",
        )

    def update_meta(
        self,
        id: int,
        data: dict,
        agent_storage_folder: str = None,
        relative_app_dir: str = None,
    ):
        """
        Update given task metadata
        :param id: int — task id
        :param data: dict — meta data to update
        """
        if type(data) == dict:
            data.update({"id": id})
            if agent_storage_folder is None and relative_app_dir is not None:
                raise ValueError(
                    "Both arguments (agent_storage_folder and relative_app_dir) has to be defined or None"
                )
            if agent_storage_folder is not None and relative_app_dir is None:
                raise ValueError(
                    "Both arguments (agent_storage_folder and relative_app_dir) has to be defined or None"
                )
            if agent_storage_folder is not None and relative_app_dir is not None:
                data["agentStorageFolder"] = {
                    "hostDir": agent_storage_folder,
                    "folder": relative_app_dir,
                }

        self._api.post("tasks.meta.update", data)

    def _update_app_content(self, task_id: int, data_patch: List[Dict] = None, state: Dict = None):
        payload = {}
        if data_patch is not None and len(data_patch) > 0:
            payload[ApiField.DATA] = data_patch
        if state is not None and len(state) > 0:
            payload[ApiField.STATE] = state

        resp = self._api.post(
            "tasks.app-v2.data.set",
            {ApiField.TASK_ID: task_id, ApiField.PAYLOAD: payload},
        )
        return resp.json()

    def set_output_error(
        self,
        task_id: int,
        title: str,
        description: Optional[str] = None,
        show_logs: Optional[bool] = True,
    ) -> Dict:
        """
        Set custom error message to the task output.

        :param task_id: Application task ID.
        :type task_id: int
        :param title: Error message to be displayed in the task output.
        :type title: str
        :param description: Description to be displayed in the task output.
        :type description: Optional[str]
        :param show_logs: If True, the link to the task logs will be displayed in the task output.
        :type show_logs: Optional[bool], default True
        :return: Response JSON.
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
               load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            task_id = 12345
            title = "Something went wrong"
            description = "Please check the task logs"
            show_logs = True
            api.task.set_output_error(task_id, title, description, show_logs)
        """

        output = {
            ApiField.GENERAL: {
                "icon": {
                    "className": "zmdi zmdi-alert-octagon",
                    "color": "#ff83a6",
                    "backgroundColor": "#ffeae9",
                },
                "title": title,
                "showLogs": show_logs,
                "isError": True,
            }
        }

        if description is not None:
            output[ApiField.GENERAL]["description"] = description

        resp = self._api.post(
            "tasks.output.set",
            {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output},
        )
        return resp.json()

    def set_output_text(
        self,
        task_id: int,
        title: str,
        description: Optional[str] = None,
        show_logs: Optional[bool] = False,
        zmdi_icon: Optional[str] = "zmdi-comment-alt-text",
        icon_color: Optional[str] = "#33c94c",
        background_color: Optional[str] = "#d9f7e4",
    ) -> Dict:
        """
        Set custom text message to the task output.

        :param task_id: Application task ID.
        :type task_id: int
        :param title: Text message to be displayed in the task output.
        :type title: str
        :param description: Description to be displayed in the task output.
        :type description: Optional[str]
        :param show_logs: If True, the link to the task logs will be displayed in the task output.
        :type show_logs: Optional[bool], default False
        :param zmdi_icon: Icon class name from Material Design Icons (ZMDI).
        :type zmdi_icon: Optional[str], default "zmdi-comment-alt-text"
        :param icon_color: Icon color in HEX format.
        :type icon_color: Optional[str], default "#33c94c" (nearest Duron Jolly Green)
        :param background_color: Background color in HEX format.
        :type background_color: Optional[str], default "#d9f7e4" (Cosmic Latte)
        :return: Response JSON.
        :rtype: Dict
        :Usage example:

        .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            task_id = 12345
            title = "Task is finished"
            api.task.set_output_text(task_id, title)
        """

        output = {
            ApiField.GENERAL: {
                "icon": {
                    "className": f"zmdi {zmdi_icon}",
                    "color": icon_color,
                    "backgroundColor": background_color,
                },
                "title": title,
                "showLogs": show_logs,
                "isError": False,
            }
        }

        if description is not None:
            output[ApiField.GENERAL]["description"] = description

        resp = self._api.post(
            "tasks.output.set",
            {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output},
        )
        return resp.json()

    def update_status(
        self,
        task_id: int,
        status: Status,
    ) -> None:
        """Sets the specified status for the task.

        :param task_id: Task ID in Supervisely.
        :type task_id: int
        :param status: Task status to set.
        :type status: One of the values from :class:`Status`, e.g. Status.FINISHED, Status.ERROR, etc.
        :raises ValueError: If the status value is not allowed.
        """
        # If status was passed without converting to string, convert it.
        # E.g. Status.FINISHED -> "finished"
        status = str(status)
        if status not in self.Status.values():
            raise ValueError(
                f"Invalid status value: {status}. Allowed values: {self.Status.values()}"
            )
        self._api.post("tasks.status.update", {ApiField.ID: task_id, ApiField.STATUS: status})

    def set_output_experiment(self, task_id: int, experiment_info: dict) -> Dict:
        """
        Sets output for the task with experiment info.

        :param task_id: Task ID in Supervisely.
        :type task_id: int
        :param experiment_info: Experiment info from TrainApp.
        :type experiment_info: dict
        :return: None
        :rtype: :class:`NoneType`

        Example of experiment_info:

            experiment_info = {
                'experiment_name': '247 Lemons RT-DETRv2-M',
                'framework_name': 'RT-DETRv2',
                'model_name': 'RT-DETRv2-M',
                'task_type': 'object detection',
                'project_id': 76,
                'project_version': {'id': 222, 'version': 4},
                'task_id': 247,
                'model_files': {'config': 'model_config.yml'},
                'checkpoints': ['checkpoints/best.pth', 'checkpoints/checkpoint0025.pth', 'checkpoints/checkpoint0050.pth', 'checkpoints/last.pth'],
                'best_checkpoint': 'best.pth',
                'export': {'ONNXRuntime': 'export/best.onnx'},
                'app_state': 'app_state.json',
                'model_meta': 'model_meta.json',
                'train_val_split': 'train_val_split.json',
                'train_size': 4,
                'val_size': 2,
                'train_collection_id': 530,
                'val_collection_id': 531,
                'hyperparameters': 'hyperparameters.yaml',
                'hyperparameters_id': 45234,
                'artifacts_dir': '/experiments/76_Lemons/247_RT-DETRv2/',
                'datetime': '2025-01-22 18:13:43',
                'experiment_report_id': 87654,
                'evaluation_report_id': 12961,
                'evaluation_report_link': 'https://app.supervisely.com/model-benchmark?id=12961',
                'evaluation_metrics': {
                    'mAP': 0.994059405940594,
                    'AP50': 1.0, 'AP75': 1.0,
                    'f1': 0.9944444444444445,
                    'precision': 0.9944444444444445,
                    'recall': 0.9944444444444445,
                    'iou': 0.9726227736959404,
                    'classification_accuracy': 1.0,
                    'calibration_score': 0.8935745942476048,
                    'f1_optimal_conf': 0.500377893447876,
                    'expected_calibration_error': 0.10642540575239527,
                    'maximum_calibration_error': 0.499622106552124
                },
                'primary_metric': 'mAP'
                'logs': {
                    'type': 'tensorboard',
                    'link': '/experiments/76_Lemons/247_RT-DETRv2/logs/'
                },
                # These fields are present only in task_info
                'project_preview': 'https://app.supervisely.com/...',
                'has_report': True,
            }
        """
        output = {
            ApiField.EXPERIMENT: {ApiField.DATA: {**experiment_info}},
        }
        resp = self._api.post(
            "tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output}
        )
        return resp.json()

    def is_running(self, task_id: int) -> bool:
        """
        Check if the task is running.

        :param task_id: Task ID in Supervisely.
        :type task_id: int
        :return: True if the task is running, False otherwise.
        :rtype: bool
        """
        try:
            self.send_request(task_id, "is_running", {}, retries=1, raise_error=True)
        except requests.exceptions.HTTPError as e:
            return False
        return True

    def is_ready(self, task_id: int) -> bool:
        """
        Check if the task is ready.

        :param task_id: Task ID in Supervisely.
        :type task_id: int
        :return: True if the task is ready, False otherwise.
        :rtype: bool
        """
        try:
            return (
                self.send_request(task_id, "is_ready", {}, retries=1, raise_error=True)["status"]
                == "ready"
            )
        except requests.exceptions.HTTPError as e:
            return False
