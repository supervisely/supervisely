# coding: utf-8
from __future__ import annotations

import json
from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, List, NamedTuple, Optional, Union

from supervisely._utils import is_community, is_development, take_with_default
from supervisely.api.module_api import ApiField
from supervisely.api.task_api import TaskApi

# from supervisely.app.constants import DATA, STATE, CONTEXT, TEMPLATE
STATE = "state"
DATA = "data"
TEMPLATE = "template"

from functools import wraps

from pkg_resources import parse_version

from supervisely import env, logger
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.file_api import FileInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.io.fs import ensure_base_path, str_is_url
from supervisely.io.json import validate_json
from supervisely.task.progress import Progress

_context_menu_targets = {
    "files_folder": {
        "help": "Context menu of folder in Team Files. Target value is directory path.",
        "type": str,
        "key": "slyFolder",
    },
    "files_file": {
        "help": "Context menu of file in Team Files. Target value is file path.",
        "type": str,
        "key": "slyFile",
    },
    "images_project": {
        "help": "Context menu of images project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "images_dataset": {
        "help": "Context menu of images dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "videos_project": {
        "help": "Context menu of videos project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "videos_dataset": {
        "help": "Context menu of videos dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_episodes_project": {
        "help": "Context menu of pointcloud episodes project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_episodes_dataset": {
        "help": "Context menu of pointcloud episodes dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "point_cloud_project": {
        "help": "Context menu of pointclouds project. Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "point_cloud_dataset": {
        "help": "Context menu of pointclouds dataset. Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "volumes_project": {
        "help": "Context menu of volumes project (DICOMs). Target value is project id.",
        "type": int,
        "key": "slyProjectId",
    },
    "volumes_dataset": {
        "help": "Context menu of volumes dataset (DICOMs). Target value is dataset id.",
        "type": int,
        "key": "slyDatasetId",
    },
    "team": {
        "help": "Context menu of team. Target value is team id.",
        "type": int,
        "key": "slyTeamId",
    },
    "team_member": {
        "help": "Context menu of team member. Target value is user id.",
        "type": int,
        "key": "slyMemberId",
    },
    "labeling_job": {
        "help": "Context menu of labeling job. Target value is labeling job id.",
        "type": int,
        "key": "slyJobId",
    },
    "ecosystem": {
        "help": "Run button in ecosystem. It is not needed to define any target",
        "key": "nothing",
    },
}

# Used to check if the instance is compatible with the workflow features
# and to avoid multiple requests to the API.
# Consists of the instance version and the result of the check for each necessary version during the session.
# Example: {"instance_version": "6.10.1", "6.9.31": True}
_workflow_compatibility_version_cache = {}


def check_workflow_compatibility(api, min_instance_version: str) -> bool:
    """Check if the instance is compatible with the workflow features.
    If the instance is not compatible, the user will be notified about it.

    :param api: Supervisely API object
    :type api: supervisely.api.api.Api
    :param min_instance_version: Minimum version of the instance that supports workflow features
    :type min_instance_version: str
    :return: True if the instance is compatible, False otherwise
    :rtype: bool
    """

    global _workflow_compatibility_version_cache
    try:
        if min_instance_version in _workflow_compatibility_version_cache:
            return _workflow_compatibility_version_cache[min_instance_version]

        instance_version = _workflow_compatibility_version_cache.setdefault(
            "instance_version", api.instance_version
        )

        if instance_version == "unknown":
            # to check again on the next call
            del _workflow_compatibility_version_cache["instance_version"]
            logger.info(
                "Can not check compatibility with Supervisely instance. "
                "Workflow features will be disabled."
            )
            return False

        is_compatible = parse_version(instance_version) >= parse_version(min_instance_version)
        _workflow_compatibility_version_cache[min_instance_version] = is_compatible

        if not is_compatible:
            message = f"Supervisely instance version '{instance_version}' does not support the following workflow features."
            if not is_community():
                message += f" To enable them, please update your instance to version '{min_instance_version}' or higher."

            logger.info(message)

        return is_compatible

    except Exception as e:
        logger.error(
            "Can not check compatibility with Supervisely instance. "
            f"Workflow features will be disabled. Error: {repr(e)}"
        )
        return False


class AppInfo(NamedTuple):
    """AppInfo"""

    id: int
    created_by_id: int
    module_id: int
    disabled: bool
    user_login: str
    config: dict
    name: str
    slug: str
    is_shared: bool
    tasks: List[Dict]
    repo: str
    team_id: int


class ModuleInfo(NamedTuple):
    """ModuleInfo in Ecosystem"""

    id: int
    slug: str
    name: str
    type: str
    config: dict
    readme: str
    repo: str
    github_repo: str
    meta: dict
    created_at: str
    updated_at: str

    @staticmethod
    def from_json(data: dict) -> ModuleInfo:
        info = ModuleInfo(
            id=data["id"],
            slug=data["slug"],
            name=data["name"],
            type=data["type"],
            config=data["config"],
            readme=data.get("readme"),
            repo=data.get("repositoryModuleUrl"),
            github_repo=data.get("repo"),
            meta=data.get("meta"),
            created_at=data["createdAt"],
            updated_at=data["updatedAt"],
        )
        if "contextMenu" in info.config:
            info.config["context_menu"] = info.config["contextMenu"]
        return info

    def get_latest_release(self, default=""):
        release = self.meta.get("releases", [default])[0]
        return release

    def arguments_help(self):
        modal_args = self.get_modal_window_arguments()
        if len(modal_args) == 0:
            print(
                f"App '{self.name}' has no additional options \n"
                "that can be configured manually in modal dialog window \n"
                "before running app."
            )
        else:
            print(
                f"App '{self.name}' has additional options "
                "that can be configured manually in modal dialog window before running app. "
                "You can change them or keep defaults: "
            )
            print(json.dumps(modal_args, sort_keys=True, indent=4))

        targets = self.get_context_menu_targets()

        if len(targets) > 0:
            print("App has to be started from the context menus:")
            for target in targets:
                print(
                    f'{target} : {_context_menu_targets.get(target, {"help": "empty description"})["help"]}'
                )
            print(
                "It is needed to call get_arguments method with defined target argument (pass one of the values above)."
            )

        if "ecosystem" in targets:
            pass

    def get_modal_window_arguments(self):
        params = self.config.get("modalTemplateState", {})
        return params

    def get_arguments(self, **kwargs) -> Dict[str, Any]:
        """Returns arguments for launching the application.
        It should be used with api.app.start() method.
        See usage example below.

        :return: arguments for launching the application
        :rtype: Dict[str, Any]
        :raises ValueError: if arguments was not passed, and the application is not
            starting from the context menu Ecosystem
        :raises KeyError: if more than one target was passed
        :raises KeyError: if invalid target was passed
        :raises ValueError: if invalid type of target value was passed
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            module_id = 81
            module_info = api.app.get_ecosystem_module_info(module_id)

            project_id = 12345
            params = module_info.get_arguments(images_project=project_id)

            # Now we can use params to start the application:
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
        params = self.config.get("modalTemplateState", {})
        targets = self.get_context_menu_targets()
        if len(targets) > 0 and len(kwargs) == 0 and "ecosystem" not in targets:
            raise ValueError(
                "target argument has to be defined. Call method 'arguments_help' to print help info for developer"
            )
        if len(kwargs) > 1:
            raise KeyError("Only one target is allowed")
        if len(kwargs) == 1:
            # params["state"] = {}
            for target_key, target_value in kwargs.items():
                if target_key not in targets:
                    raise KeyError(
                        f"You passed {target_key}, but allowed only one of the targets: {targets}"
                    )
                key = _context_menu_targets[target_key]["key"]
                valid_type = _context_menu_targets[target_key]["type"]
                if type(target_value) is not valid_type:
                    raise ValueError(
                        f"Target {target_key} has value {target_value} of type {type(target_value)}. Allowed type is {valid_type}"
                    )
                params[key] = target_value
        return params

    def get_context_menu_targets(self):
        if "context_menu" in self.config:
            if "target" in self.config["context_menu"]:
                return self.config["context_menu"]["target"]
        return []


class SessionInfo(NamedTuple):
    """SessionInfo"""

    task_id: int
    user_id: int
    module_id: int  # in ecosystem
    app_id: int  # in team (recent apps)

    details: dict

    @staticmethod
    def from_json(data: dict) -> SessionInfo:
        # {'taskId': 21012, 'userId': 6, 'moduleId': 83, 'appId': 578}

        if "meta" in data:
            info = SessionInfo(
                task_id=data["id"],
                user_id=data["createdBy"],
                module_id=data["moduleId"],
                app_id=data["meta"]["app"]["id"],
                details=data,
            )
        else:
            info = SessionInfo(
                task_id=data["taskId"],
                user_id=data["userId"],
                module_id=data["moduleId"],
                app_id=data["appId"],
                details={},
            )
        return info


@dataclass
class WorkflowSettings:
    """Used to customize the appearance and behavior of the workflow node.

    :param title: Title of the node. It is displayed in the node header.
                  Title is formatted with the `<h4>` tag.
    :type title: Optional[str]
    :param icon: Icon of the node. It is displayed in the node body.
                 The icon name should be from the Material Design Icons set.
                 Do not include the 'zmdi-' prefix.
    :type icon: Optional[str]
    :param icon_color: Color of the icon in hexadecimal format.
    :type icon_color: Optional[str]
    :param icon_bg_color: Background color of the icon in hexadecimal format.
    :type icon_bg_color: Optional[str]
    :param url: URL to be opened when the user clicks on it. Must start with a slash and be relative to the instance.
    :type url: Optional[str]
    :param url_title: Title of the URL.
    :type url_title: Optional[str]
    :param description: Description of the node. It is displayed under the title line.
                        It's not recommended to use it for long texts.
                        Description is formatted with the `<small>` tag and used to clarify specific information.
    :type description: Optional[str]
    """

    title: Optional[str] = None
    icon: Optional[str] = None
    icon_color: Optional[str] = None
    icon_bg_color: Optional[str] = None
    url: Optional[str] = None
    url_title: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if (self.url and not self.url_title) or (not self.url and self.url_title):
            logger.info(
                "Workflow Warning: both 'url' and 'url_title' must be set together in WorkflowSettings. "
                "Setting MainLink to default."
            )
            self.url = None
            self.url_title = None
        if not all([self.icon, self.icon_color, self.icon_bg_color]) and any(
            [self.icon, self.icon_color, self.icon_bg_color]
        ):
            logger.info(
                "Workflow Warning: all three parameters 'icon', 'icon_color', and 'icon_bg_color' must be set together in WorkflowSettings. "
                "Setting Icon to default."
            )
            self.icon = None
            self.icon_color = None
            self.icon_bg_color = None

    @property
    def as_dict(self) -> Dict[str, Any]:
        result = {}
        if self.title is not None:
            result["title"] = f"<h4>{self.title}</h4>"
        if self.description is not None:
            result["description"] = f"<small>{self.description}</small>"
        if self.icon is not None and self.icon_color is not None and self.icon_bg_color is not None:
            result["icon"] = {}
            result["icon"]["icon"] = f"zmdi-{self.icon}"
            result["icon"]["color"] = self.icon_color
            result["icon"]["backgroundColor"] = self.icon_bg_color
        if self.url is not None and self.url_title is not None:
            result["mainLink"] = {}
            result["mainLink"]["url"] = self.url
            result["mainLink"]["title"] = self.url_title
        return result


@dataclass
class WorkflowMeta:
    """Used to customize the appearance of the workflow main and/or relation node.

    :param relation_settings: customizes the appearance of the relation node - inputs and outputs
    :type relation_settings: Optional[WorkflowSettings]
    :param node_settings: customizes the appearance of the main node - the task itself
    :type node_settings: Optional[WorkflowSettings]
    """

    relation_settings: Optional[WorkflowSettings] = None
    node_settings: Optional[WorkflowSettings] = None

    def __post_init__(self):
        if not (self.relation_settings or self.node_settings):
            logger.info(
                "Workflow Warning: at least one of 'relation_settings' or 'node_settings' must be specified in WorkflowMeta. "
                "Customization will not be applied."
            )

    @property
    def as_dict(self) -> Dict[str, Any]:
        result = {}
        if self.relation_settings is not None:
            result["customRelationSettings"] = self.relation_settings.as_dict
        if self.node_settings is not None:
            result["customNodeSettings"] = self.node_settings.as_dict
        return result if result != {} else None

    @classmethod
    def create_as_dict(cls, **kwargs) -> Dict[str, Any]:
        instance = cls(**kwargs)
        return instance.as_dict


class AppApi(TaskApi):
    """AppApi"""

    class Workflow:
        """The workflow functionality is used to create connections between the states of projects and tasks (application sessions) that interact with them in some way.
        By assigning connections to various entities, the workflow tab allows tracking the history of project changes.
        The active task always acts as a node, for which input and output elements are defined.
        There can be multiple input and output elements.
        A task can also be used as an input or output element.
        For example, an inference task takes a deployed model and a project as inputs, and the output is a new state of the project.
        This functionality uses versioning optionally.

        If instances are not compatible with the workflow features, the functionality will be disabled.

        :param api: Supervisely API object
        :type api: supervisely.api.api.Api
        :param min_instance_version: Minimum version of the instance that supports workflow features
        :type min_instance_version: str
        """

        __custom_meta_schema = {
            "type": "object",
            "definitions": {
                "settings": {
                    "type": "object",
                    "properties": {
                        "icon": {
                            "type": "object",
                            "properties": {
                                "icon": {"type": "string"},
                                "color": {"type": "string"},
                                "backgroundColor": {"type": "string"},
                            },
                            "required": ["icon", "color", "backgroundColor"],
                            "additionalProperties": False,
                        },
                        "title": {"type": "string"},  # html
                        "description": {"type": "string"},  # html
                        "mainLink": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
                            "required": ["url", "title"],
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                }
            },
            "properties": {
                "customRelationSettings": {"$ref": "#/definitions/settings"},
                "customNodeSettings": {"$ref": "#/definitions/settings"},
            },
            "additionalProperties": False,
            "anyOf": [
                {"required": ["customRelationSettings"]},
                {"required": ["customNodeSettings"]},
            ],
        }

        def __init__(self, api):
            self._api = api
            # minimum instance version that supports workflow features
            self._min_instance_version = "6.9.31"
            # for development purposes
            self._enabled = True
            if is_development():
                self._enabled = False
                logger.warning(
                    "Workflow is disabled in development mode. "
                    "To enable it, use `api.app.workflow.enable()` right after Api initialization."
                )

        def enable(self):
            """Enable the workflow functionality."""
            self._enabled = True
            logger.info("Workflow is enabled.")

        def disable(self):
            """Disable the workflow functionality."""
            self._enabled = False
            logger.info("Workflow is disabled.")

        def is_enabled(self) -> bool:
            """Check if the workflow functionality is enabled."""
            logger.info(f"Workflow check: is {'enabled' if self._enabled else 'disabled'}.")
            return self._enabled

        # pylint: disable=no-self-argument
        def check_instance_compatibility(min_instance_version: Optional[str] = None):
            """Decorator to check instance compatibility with workflow features.
            If the instance is not compatible, the function will not be executed.

            :param min_instance_version: Determine the minimum instance version that accepts the workflow method.
            If not specified, the minimum version will be "6.9.31".
            :type min_instance_version: Optional[str]
            """

            def decorator(func):
                @wraps(func)
                def wrapper(self, *args, **kwargs):
                    version_to_check = (
                        min_instance_version
                        if min_instance_version is not None
                        else self._min_instance_version
                    )
                    if (
                        not check_workflow_compatibility(self._api, version_to_check)
                        or not self._enabled
                    ):
                        logger.info(f"Workflow method `{func.__name__}` is disabled.")
                        return
                    return func(self, *args, **kwargs)

                return wrapper

            return decorator

        # pylint: enable=no-self-argument

        def _add_edge(
            self,
            data: dict,
            transaction_type: str,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input or output to a workflow node.

            :param data: Data to be added to the workflow node.
            :type data: dict
            :param transaction_type: Type of transaction "input" or "output".
            :type transaction_type: str
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: dict
            """
            try:
                if task_id is None:
                    node_id = self._api.task_id
                else:
                    node_id = task_id
                if node_id is None:
                    raise ValueError(
                        "Task ID cannot be automatically determined. Please specify it manually."
                    )
                node_type = "task"
                if not getattr(self, "team_id", None) and node_id:
                    self.team_id = self._api.task.get_info_by_id(node_id).get(ApiField.TEAM_ID)
                if not self.team_id:
                    raise ValueError("Failed to get Team ID")
                api_endpoint = f"workflow.node.add-{transaction_type}"
                data_type = data.get("data_type")
                data_id = data.get("data_id") if data_type != "app_session" else node_id
                data_meta = data.get("meta", {})
                if meta is not None:
                    if isinstance(meta, WorkflowMeta):
                        meta = meta.as_dict
                    if validate_json(meta, self.__custom_meta_schema):
                        data_meta.update(meta)
                    else:
                        logger.warn("Invalid customization meta, will not be added to the node.")
                payload = {
                    ApiField.TEAM_ID: self.team_id,
                    ApiField.NODE: {ApiField.TYPE: node_type, ApiField.ID: node_id},
                    ApiField.TYPE: data_type,
                }
                if data_id:
                    payload[ApiField.ID] = data_id
                if data_meta:
                    payload[ApiField.META] = data_meta
                response = self._api.post(api_endpoint, payload)
                return response.json()
            except Exception:
                logger.error(
                    f"Failed to add {transaction_type} node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_input_project(
            self,
            project: Optional[Union[int, ProjectInfo]] = None,
            version_id: Optional[int] = None,
            version_num: Optional[int] = None,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "project" to the workflow node.
            The project version can be specified to indicate that the project version was used especially for this task.
            Arguments project and version_id are mutually exclusive. If both are specified, version_id will be used.
            Argument version_num can only be used in conjunction with the project.
            This type is used to show that the application has used the specified project.
            Customization of the project node is not supported and will be ignored.
            You can only customize the main node with this method.

            :param project: Project ID or ProjectInfo object.
            :type project: Optional[Union[int, ProjectInfo]]
            :param version_id: Version ID of the project.
            :type version_id: Optional[int]
            :param version_num: Version number of the project. This argument can only be used in conjunction with the project.
            :type version_num: Optional[int]
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                if project is None and version_id is None and version_num is None:
                    raise ValueError(
                        "At least one of project, version_id or version_num must be specified"
                    )
                if version_id is not None and version_num is not None:
                    raise ValueError("Only one of version_id or version_num can be specified")
                if project is None and version_num is not None:
                    raise ValueError(
                        "Argument version_num cannot be used without specifying a project argument"
                    )
                data_type = "project"
                data_id = None
                if isinstance(project, ProjectInfo):
                    data_id = project.id
                elif isinstance(project, int):
                    data_id = project
                if version_num:
                    version_id = self._api.project.version.get_id_by_number(data_id, version_num)
                if version_id:
                    data_id = version_id
                    data_type = "project_version"
                data = {
                    "data_type": data_type,
                    "data_id": data_id,
                }
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input project node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_input_dataset(
            self,
            dataset: Union[int, DatasetInfo],
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "dataset" to the workflow node.
            This type is used to show that the application has used the specified dataset.
            Customization of the dataset node is not supported and will be ignored.
            You can only customize the main node with this method.

            :param dataset: Dataset ID or DatasetInfo object.
            :type dataset: Union[int, DatasetInfo]
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "dataset"
                if isinstance(dataset, DatasetInfo):
                    dataset = dataset.id
                data = {"data_type": data_type, "data_id": dataset}
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input dataset node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_input_file(
            self,
            file: Union[int, FileInfo, str],
            model_weight: bool = False,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "file" to the workflow node.
            This type is used to show that the application has used the specified file.

            :param file: File ID, FileInfo object or file path in team Files.
            :type file: Union[int, FileInfo, str]
            :param model_weight: Flag to indicate if the file is a model weight.
            :type model_weight: bool
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data = {}
                data_type = "file"
                if isinstance(file, FileInfo):
                    file_id = file.id
                elif isinstance(file, int):
                    file_id = file
                elif isinstance(file, str):
                    if str_is_url(file):
                        raise NotImplementedError("URLs are not supported yet")
                    file_id = self._api.file.get_info_by_path(env.team_id(), file).id
                else:
                    raise ValueError(f"Invalid file type: {type(file)}")
                if model_weight:
                    data_type = "model_weight"
                data["data_type"] = data_type
                data["data_id"] = file_id
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input file node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_input_folder(
            self,
            path: str,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "folder" to the workflow node.
            Path to the folder is a path in Team Files.
            This type is used to show that the application has used files from the specified folder.

            :param path: Path to the folder in Team Files.
            :type path: str
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                from pathlib import Path

                if not path.startswith("/"):
                    path = "/" + path
                try:
                    Path(path)
                except Exception as e:
                    raise ValueError(f"The provided string '{path}' is not a valid path: {str(e)}")
                data_type = "folder"
                data = {"data_type": data_type, "data_id": path}
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input folder node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_input_task(
            self,
            input_task_id: int,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "task" to the workflow node.
            This type usually indicates that the one application has used another application for its work.

            :param input_task_id: Task ID that is used as input.
            :type input_task_id: int
            :param task_id: Task ID of the node. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "task"
                data = {"data_type": data_type, "data_id": input_task_id}
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input task node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        # pylint: disable=redundant-keyword-arg
        @check_instance_compatibility(
            min_instance_version="6.11.11"
        )  # Min instance version that accepts this method
        def add_input_job(
            self,
            id: int,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add input type "job" to the workflow node. Job is a Labeling Job.
            This type indicates that the application has utilized a labeling job during its operation.

            :param id: Labeling Job ID.
            :type id: int
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "job"
                data = {"data_type": data_type, "data_id": id}
                return self._add_edge(data, "input", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add input job node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        # pylint: enable=redundant-keyword-arg

        @check_instance_compatibility()
        def add_output_project(
            self,
            project: Union[int, ProjectInfo],
            version_id: Optional[int] = None,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "project" to the workflow node.
            The project version can be specified with "version" argument to indicate that the project version was created especially as result of this task.
            This type is used to show that the application has created a project with the result of its work.
            Customization of the project node is not supported and will be ignored.
            You can only customize the main node with this method.

            :param project: Project ID or ProjectInfo object.
            :type project: Union[int, ProjectInfo]
            :param version_id: Version ID of the project.
            :type version_id: Optional[int]
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                if project is None and version_id is None:
                    raise ValueError("Project or version must be specified")
                data_type = "project"
                data_id = None
                if isinstance(project, ProjectInfo):
                    data_id = project.id
                elif isinstance(project, int):
                    data_id = project
                if version_id:
                    data_id = version_id
                    data_type = "project_version"
                data = {
                    "data_type": data_type,
                    "data_id": data_id,
                }
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output project node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_output_dataset(
            self,
            dataset: Union[int, DatasetInfo],
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "dataset" to the workflow node.
            This type is used to show that the application has created a dataset with the result of its work.
            Customization of the dataset node is not supported and will be ignored.
            You can only customize the main node with this method.

            :param dataset: Dataset ID or DatasetInfo object.
            :type dataset: Union[int, DatasetInfo]
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "dataset"
                if isinstance(dataset, DatasetInfo):
                    dataset = dataset.id
                data = {"data_type": data_type, "data_id": dataset}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output dataset node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_output_file(
            self,
            file: Union[int, FileInfo],
            model_weight=False,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "file" to the workflow node.
            This type is used to show that the application has created a file with the result of its work.

            :param file: File ID or FileInfo object.
            :type file: Union[int, FileInfo]
            :param model_weight: Flag to indicate if the file is a model weight.
            :type model_weight: bool
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "file"
                if isinstance(file, FileInfo):
                    file = file.id
                if model_weight:
                    data_type = "model_weight"
                data = {"data_type": data_type, "data_id": file}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output file node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_output_folder(
            self,
            path: str,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "folder" to the workflow node.
            Path to the folder is a path in Team Files.
            This type is used to show that the application has created a folder with the result files of its work.

            :param path: Path to the folder.
            :type path: str
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                from pathlib import Path

                if not path.startswith("/"):
                    path = "/" + path
                try:
                    Path(path)
                except Exception as e:
                    raise ValueError(f"The provided string '{path}' is not a valid path: {str(e)}")
                data_type = "folder"
                data = {"data_type": data_type, "data_id": path}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output folder node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_output_app(
            self,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "app_session" to the workflow node.
            This type is used to show that the application has an offline session in which you can find the result of its work.

            :param task_id: App Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "app_session"
                data = {"data_type": data_type}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output app node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        @check_instance_compatibility()
        def add_output_task(
            self,
            output_task_id: int,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "task" to the workflow node.
            This type is used to show that the application has created a task with the result of its work.

            :param output_task_id: Created task ID.
            :type output_task_id: int
            :param task_id: Task ID of the node. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "task"
                data = {"data_type": data_type, "data_id": output_task_id}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output task node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        # pylint: disable=redundant-keyword-arg
        @check_instance_compatibility(
            min_instance_version="6.11.11"
        )  # Min instance version that accepts this method
        def add_output_job(
            self,
            id: int,
            task_id: Optional[int] = None,
            meta: Optional[Union[WorkflowMeta, dict]] = None,
        ) -> dict:
            """
            Add output type "job" to the workflow node. Job is a Labeling Job.
            This type is used to show that the application has created a labeling job with the result of its work.

            :param id: Labeling Job ID.
            :type id: int
            :param task_id: Task ID. If not specified, the task ID will be determined automatically.
            :type task_id: Optional[int]
            :param meta: Additional data for node customization.
            :type meta: Optional[Union[WorkflowMeta, dict]]
            :return: Response from the API.
            :rtype: :class:`dict`
            """
            try:
                data_type = "job"
                data = {"data_type": data_type, "data_id": id}
                return self._add_edge(data, "output", task_id, meta)
            except Exception:
                logger.error(
                    "Failed to add output job node to the workflow "
                    "(this error will not interrupt other code execution)."
                )
                return {}

        # pylint: enable=redundant-keyword-arg

    def __init__(self, api):
        super().__init__(api)
        self.workflow = self.Workflow(api)

    @staticmethod
    def info_sequence():
        """info_sequence"""
        return [
            ApiField.ID,
            ApiField.CREATED_BY_ID,
            ApiField.MODULE_ID,
            ApiField.DISABLED,
            ApiField.USER_LOGIN,
            ApiField.CONFIG,
            ApiField.NAME,
            ApiField.SLUG,
            ApiField.IS_SHARED,
            ApiField.TASKS,
            ApiField.REPO,
            ApiField.TEAM_ID,
        ]

    @staticmethod
    def info_tuple_name():
        """info_tuple_name"""
        return "AppInfo"

    def _convert_json_info(self, info: dict, skip_missing=True) -> AppInfo:
        """_convert_json_info"""
        res = super(TaskApi, self)._convert_json_info(info, skip_missing=skip_missing)
        return AppInfo(**res._asdict())

    def get_info_by_id(self, id: int) -> AppInfo:
        """
        :param id: int
        :return: application info by numeric id
        """
        return self._get_info_by_id(id, "apps.info")

    def get_list(
        self,
        team_id,
        filter=None,
        context=None,
        repository_key=None,
        show_disabled=False,
        integrated_into=None,
        session_tags=None,
        only_running=False,
        with_shared=True,
    ) -> List[AppInfo]:
        """get_list"""

        return self.get_list_all_pages(
            method="apps.list",
            data={
                "teamId": team_id,
                "filter": take_with_default(
                    filter, []
                ),  # for example [{"field": "id", "operator": "=", "value": None}]
                "context": take_with_default(context, []),  # for example ["images_project"]
                "repositoryKey": repository_key,
                "integratedInto": take_with_default(
                    integrated_into, []
                ),  # for example ["image_annotation_tool"]
                "sessionTags": take_with_default(session_tags, []),  # for example ["string"]
                "onlyRunning": only_running,
                "showDisabled": show_disabled,
                "withShared": with_shared,
            },
        )

    def run_dtl(self, workspace_id, dtl_graph, agent_id=None):
        """run_dtl"""
        raise RuntimeError("Method is unavailable")

    def _run_plugin_task(
        self,
        task_type,
        agent_id,
        plugin_id,
        version,
        config,
        input_projects,
        input_models,
        result_name,
    ):
        """_run_plugin_task"""
        raise RuntimeError("Method is unavailable")

    def run_train(
        self,
        agent_id,
        input_project_id,
        input_model_id,
        result_nn_name,
        train_config=None,
    ):
        """run_train"""
        raise RuntimeError("Method is unavailable")

    def run_inference(
        self,
        agent_id,
        input_project_id,
        input_model_id,
        result_project_name,
        inference_config=None,
    ):
        """run_inference"""
        raise RuntimeError("Method is unavailable")

    def get_training_metrics(self, task_id):
        """get_training_metrics"""
        raise RuntimeError("Method is unavailable")

    def deploy_model(self, agent_id, model_id):
        """deploy_model"""
        raise RuntimeError("Method is unavailable")

    def get_import_files_list(self, id):
        """get_import_files_list"""
        raise RuntimeError("Method is unavailable")

    def download_import_file(self, id, file_path, save_path):
        """download_import_file"""
        raise RuntimeError("Method is unavailable")

    def create_task_detached(self, workspace_id, task_type: str = None):
        """create_task_detached"""
        raise RuntimeError("Method is unavailable")

    def upload_files(self, task_id, abs_paths, names, progress_cb=None):
        """upload_files"""
        raise RuntimeError("Method is unavailable")

    def initialize(self, task_id, template, data=None, state=None):
        """initialize"""
        d = take_with_default(data, {})
        if "notifyDialog" not in d:
            d["notifyDialog"] = None
        if "scrollIntoView" not in d:
            d["scrollIntoView"] = None

        s = take_with_default(state, {})
        fields = [
            {"field": TEMPLATE, "payload": template},
            {"field": DATA, "payload": d},
            {"field": STATE, "payload": s},
        ]
        resp = self._api.task.set_fields(task_id, fields)
        return resp

    def get_url(self, task_id):
        """get_url"""
        return f"/apps/sessions/{task_id}"

    def download_git_file(self, app_id, version, file_path, save_path):
        """download_git_file"""
        raise NotImplementedError()

    def download_git_archive(
        self,
        ecosystem_item_id,
        app_id,
        version,
        save_path,
        log_progress=True,
        ext_logger=None,
    ):
        # pylint: disable=possibly-used-before-assignment
        """download_git_archive"""
        payload = {
            ApiField.ECOSYSTEM_ITEM_ID: ecosystem_item_id,
            ApiField.VERSION: version,
            "isArchive": True,
        }
        if app_id is not None:
            payload[ApiField.APP_ID] = app_id

        response = self._api.post("ecosystem.file.download", payload, stream=True)
        if log_progress:
            if ext_logger is None:
                ext_logger = logger

            length = None
            # Content-Length
            if "Content-Length" in response.headers:
                length = int(response.headers["Content-Length"])
            progress = Progress("Downloading: ", length, ext_logger=ext_logger, is_size=True)

        mb1 = 1024 * 1024
        ensure_base_path(save_path)
        with open(save_path, "wb") as fd:
            log_size = 0
            for chunk in response.iter_content(chunk_size=mb1):
                fd.write(chunk)
                log_size += len(chunk)
                if log_progress and log_size > mb1:
                    progress.iters_done_report(log_size)
                    log_size = 0

    def get_info(self, module_id, version=None):
        """get_info"""
        data = {ApiField.ID: module_id}
        if version is not None:
            data[ApiField.VERSION] = version
        response = self._api.post("ecosystem.info", data)
        return response.json()

    def get_ecosystem_module_info(
        self, module_id: int, version: Optional[str] = None
    ) -> ModuleInfo:
        """Returns ModuleInfo object by module id and version.

        :param module_id: ID of the module
        :type module_id: int
        :param version: version of the module, e.g. "v1.0.0"
        :type version: Optional[str]
        :return: ModuleInfo object
        :rtype: ModuleInfo
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            module_id = 81
            module_info = api.app.get_ecosystem_module_info(module_id)
        """
        data = {ApiField.ID: module_id}
        if version is not None:
            data[ApiField.VERSION] = version
        response = self._api.post("ecosystem.info", data)
        return ModuleInfo.from_json(response.json())

    def get_ecosystem_module_id(self, slug: str) -> int:
        """Returns ecosystem module id by slug.
        E.g. slug = "supervisely-ecosystem/export-to-supervisely-format".
        Slug can be obtained from the application URL in browser.

        :param slug: module slug, starts with "supervisely-ecosystem/"
        :type slug: str
        :return: ID of the module
        :rtype: int
        :raises KeyError: if module with given slug not found
        :raises KeyError: if there are multiple modules with the same slug
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            slug = "supervisely-ecosystem/export-to-supervisely-format"
            module_id = api.app.get_ecosystem_module_id(slug)
            print(f"Module {slug} has id {module_id}")
            # Module supervisely-ecosystem/export-to-supervisely-format has id 81
        """
        modules = self.get_list_all_pages(
            method="ecosystem.list",
            data={"filter": [{"field": "slug", "operator": "=", "value": slug}]},
            convert_json_info_cb=lambda x: x,
        )
        if len(modules) == 0:
            raise KeyError(f"Module {slug} not found in ecosystem")
        if len(modules) > 1:
            raise KeyError(
                f"Ecosystem is broken: there are {len(modules)} modules with the same slug: {slug}. Please, contact tech support"
            )
        return modules[0]["id"]

    def get_list_ecosystem_modules(self):
        modules = self.get_list_all_pages(
            method="ecosystem.list",
            data={},
            convert_json_info_cb=lambda x: x,
        )
        if len(modules) == 0:
            raise KeyError("No modules found in ecosystem")
        return modules

    # def get_sessions(self, workspace_id: int, filter_statuses: List[TaskApi.Status] = None):
    #     filters = [{"field": "type", "operator": "=", "value": "app"}]
    #     # filters = []
    #     if filter_statuses is not None:
    #         s = [str(status) for status in filter_statuses]
    #         filters.append({"field": "status", "operator": "in", "value": s})
    #     result = self._api.task.get_list(workspace_id=workspace_id, filters=filters)
    #     return result

    def get_sessions(
        self,
        team_id,
        module_id,
        # only_running=False,
        show_disabled=False,
        session_name=None,
        statuses: List[TaskApi.Status] = None,
    ) -> List[SessionInfo]:
        infos_json = self.get_list_all_pages(
            method="apps.list",
            data={
                "teamId": team_id,
                "filter": [{"field": "moduleId", "operator": "=", "value": module_id}],
                # "onlyRunning": only_running,
                "showDisabled": show_disabled,
            },
            convert_json_info_cb=lambda x: x,
            # validate_total=False,
        )
        if len(infos_json) == 0:
            # raise KeyError(f"App [module_id = {module_id}] not found in team {team_id}")
            return []
        if len(infos_json) > 1:
            raise KeyError(
                f"Apps list in team is broken: app [module_id = {module_id}] added to team {team_id} multiple times"
            )
        dev_tasks = []
        sessions = infos_json[0]["tasks"]

        str_statuses = []
        if statuses is not None:
            for s in statuses:
                str_statuses.append(str(s))

        for session in sessions:
            to_add = True
            if session_name is not None and session["meta"]["name"] != session_name:
                to_add = False
            if statuses is not None and session["status"] not in str_statuses:
                to_add = False
            if to_add is True:
                session["moduleId"] = module_id
                dev_tasks.append(SessionInfo.from_json(session))
        return dev_tasks

    def start(
        self,
        agent_id,
        app_id=None,
        workspace_id=None,
        description="",
        params=None,
        log_level="info",
        users_id=None,
        app_version=None,
        is_branch=False,
        task_name="run-from-python",
        restart_policy="never",
        proxy_keep_url=False,
        module_id=None,
        redirect_requests={},
    ) -> SessionInfo:
        users_ids = None
        if users_id is not None:
            users_ids = [users_id]

        new_params = {}
        if "state" not in params:
            new_params["state"] = params
        else:
            new_params = params

        if app_version is None:
            module_info = self.get_ecosystem_module_info(module_id)
            app_version = module_info.get_latest_release().get("version", "")

        result = self._api.task.start(
            agent_id=agent_id,
            app_id=app_id,
            workspace_id=workspace_id,
            description=description,
            params=new_params,
            log_level=log_level,
            users_ids=users_ids,
            app_version=app_version,
            is_branch=is_branch,
            task_name=task_name,
            restart_policy=restart_policy,
            proxy_keep_url=proxy_keep_url,
            module_id=module_id,
            redirect_requests=redirect_requests,
        )
        if type(result) is not list:
            result = [result]
        if len(result) != 1:
            raise ValueError(f"{len(result)} tasks started instead of one")
        return SessionInfo.from_json(result[0])

    def wait(
        self,
        id: int,
        target_status: TaskApi.Status,
        attempts: Optional[int] = None,
        attempt_delay_sec: Optional[int] = None,
    ):
        """wait"""
        return self._api.task.wait(
            id=id,
            target_status=target_status,
            wait_attempts=attempts,
            wait_attempt_timeout_sec=attempt_delay_sec,
        )

    def stop(self, id: int) -> TaskApi.Status:
        """stop"""
        return self._api.task.stop(id)

    def get_status(self, task_id: int) -> TaskApi.Status:
        return self._api.task.get_status(task_id)

    def is_ready_for_api_calls(self, task_id: int) -> bool:
        """
        Checks if app is ready for API calls.
        :param task_id: ID of the running task.
        :type task_id: int
        :return: True if app is ready for API calls, False otherwise.
        """
        try:
            info = self._api.app.send_request(
                task_id, "is_running", {}, timeout=1, retries=1, raise_error=True
            )
            is_running = info.get("running", False)
            if is_running:
                logger.debug(f"App {task_id} is ready for API calls")
                return True
            return False
        except:
            logger.debug(f"App {task_id} is not ready for API calls yet")
            return False

    def wait_until_ready_for_api_calls(
        self, task_id: int, attempts: int = 10, attempt_delay_sec: Optional[int] = 10
    ) -> bool:
        """
        Waits until app is ready for API calls.

        :param task_id: ID of the running task.
        :type task_id: int
        :param attempts: Number of attempts to check if app is ready for API calls.
        :type attempts: int
        :param attempt_delay_sec: Delay between attempts in seconds.
        :type attempt_delay_sec: int
        :return: True if app is ready for API calls, False otherwise.
        """
        is_ready = False
        logger.info("Waiting for app to be ready for API calls")
        for attempt in range(attempts):
            is_ready = self.is_ready_for_api_calls(task_id)
            if not is_ready:
                sleep(attempt_delay_sec)
            else:
                is_ready = True
                break
        return is_ready


# info about app in team
# {
#     "id": 7,
#     "createdBy": 1,
#     "moduleId": 16,
#     "disabled": false,
#     "userLogin": "admin",
#     "config": {
#         "icon": "https://user-images.githubusercontent.com/12828725/182186256-5ee663ad-25c7-4a62-9af1-fbfdca715b57.png",
#         "author": {"name": "Maxim Kolomeychenko"},
#         "poster": "https://user-images.githubusercontent.com/12828725/182181033-d0d1a690-8388-472e-8862-e0cacbd4f082.png",
#         "needGPU": false,
#         "headless": true,
#         "categories": ["development"],
#         "lastCommit": "96eca85e1fbed45d59db405b17c04f4d920c6c81",
#         "description": "Demonstrates how to turn your python script into Supervisely App",
#         "main_script": "src/main.py",
#         "sessionTags": [],
#         "taskLocation": "workspace_tasks",
#         "defaultBranch": "master",
#         "isPrivateRepo": false,
#         "restartPolicy": "never",
#         "slyModuleInfo": {"baseSlug": "supervisely-ecosystem/hello-world-app"},
#         "communityAgent": true,
#         "iconBackground": "#FFFFFF",
#         "integratedInto": ["standalone"],
#         "storeDataOnAgent": false,
#     },
#     "name": "Hello World!",
#     "slug": "supervisely-ecosystem/hello-world-app",
#     "moduleDisabled": false,
#     "provider": "sly_gitea",
#     "repositoryId": 42,
#     "pathPrefix": "",
#     "baseUrl": null,
#     "isShared": false,
#     "tasks": [
#         {
#             "id": 19107,
#             "type": "app",
#             "size": "0",
#             "status": "finished",
#             "startedAt": "2022-08-04T14:59:45.797Z",
#             "finishedAt": "2022-08-04T14:59:49.793Z",
#             "meta": {
#                 "app": {
#                     "id": 7,
#                     "name": "Hello World!",
#                     "version": "v1.0.4",
#                     "isBranch": false,
#                     "logLevel": "info",
#                 },
#                 "name": "",
#                 "params": {"state": {}},
#                 "hasLogs": true,
#                 "logsCnt": 60,
#                 "hasMetrics": false,
#                 "sessionToken": "PDVBF6ecX09FY75n7ufa8q_MTB28XI6XIMcJ1md4ogeN0FLTbIZyC91Js_9YkGpUQhQbCYyTE8Q=",
#                 "restartPolicy": "never",
#             },
#             "attempt": 1,
#             "archived": false,
#             "nodeId": 1,
#             "createdBy": 6,
#             "teamId": 7,
#             "description": "",
#             "isShared": false,
#             "user": "max",
#         }
#     ],
#     "repo": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "repoKey": "supervisely-ecosystem/hello-world-app",
#     "githubModuleUrl": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "repositoryModuleUrl": "https://github.com/supervisely-ecosystem/hello-world-app",
#     "teamId": 7,
# }


# infor about module in ecosystem
