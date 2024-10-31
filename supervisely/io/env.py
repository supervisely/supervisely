# coding: utf-8
import os
from typing import Callable, List, Literal, Optional, Union

RAISE_IF_NOT_FOUND = True


def flag_from_env(s: str) -> bool:
    """Returns True if passed string is a flag, False otherwise.
    Possible values to set the flag to True:
        - "1"
        - "true"
        - "yes"

    :param s: string to check
    :type s: str
    :return: True if passed string is a flag, False otherwise
    :rtype: bool
    """
    return s.upper() in ["TRUE", "YES", "1"]


def remap_gpu_devices(in_device_ids):
    """
    Working limitation for CUDA
    :param in_device_ids: real GPU devices indexes. e.g.: [3, 4, 7]
    :return: CUDA ordered GPU indexes, e.g.: [0, 1, 2]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, in_device_ids))
    return list(range(len(in_device_ids)))


def _int_from_env(value):
    if value is None:
        return value
    return int(value)


def _parse_from_env(
    name: str,
    keys: List[str],
    postprocess_fn: Callable,
    default=None,
    raise_not_found=False,
):
    for k in keys:
        if k in os.environ:
            return postprocess_fn(os.environ[k])

    # env not found
    if raise_not_found is True:
        raise KeyError(
            f"{name} is not defined as environment variable. One of the envs has to be defined: {keys}. Learn more in developer portal: https://developer.supervise.ly/getting-started/environment-variables"
        )

    return default


def agent_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns agent id from environment variable using following keys:
        - AGENT_ID

    :param raise_not_found: if True, raises KeyError if agent id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: agent id
    :rtype: int
    """
    return _parse_from_env(
        name="agent_id",
        keys=["AGENT_ID"],
        postprocess_fn=_int_from_env,
        default=None,
        raise_not_found=raise_not_found,
    )


def agent_storage(raise_not_found: Optional[bool] = True) -> str:
    """Returns path to the agent's storafe from environment variable using following keys:
        - AGENT_STORAGE

    :param raise_not_found: if True, raises KeyError if agent storage is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the agent's storage
    :rtype: str
    """
    return _parse_from_env(
        name="agent_storage",
        keys=["AGENT_STORAGE"],
        postprocess_fn=lambda x: x,
        default=None,
        raise_not_found=raise_not_found,
    )


def team_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns team id from environment variable using following keys:
        - TEAM_ID
        - CONTEXT_TEAMID
        - context.teamId

    :param raise_not_found: if True, raises KeyError if team id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: team id
    :rtype: int
    """
    return _parse_from_env(
        name="team_id",
        keys=["CONTEXT_TEAMID", "context.teamId", "TEAM_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def workspace_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns workspace id from environment variable using following keys:
        - WORKSPACE_ID
        - CONTEXT_WORKSPACEID
        - context.workspaceId

    :param raise_not_found: if True, raises KeyError if workspace id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: workspace id
    :rtype: int
    """
    return _parse_from_env(
        name="workspace_id",
        keys=["CONTEXT_WORKSPACEID", "context.workspaceId", "WORKSPACE_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def project_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns project id from environment variable using following keys:
        - PROJECT_ID
        - CONTEXT_PROJECTID
        - context.projectId
        - modal.state.slyProjectId
        - modal.state.inputProjectId

    :param raise_not_found: if True, raises KeyError if project id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: project id
    :rtype: int
    """
    return _parse_from_env(
        name="project_id",
        keys=[
            "CONTEXT_PROJECTID",
            "context.projectId",
            "modal.state.slyProjectId",
            "PROJECT_ID",
            "modal.state.inputProjectId",
        ],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def dataset_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns dataset id from environment variable using following keys:
        - DATASET_ID
        - CONTEXT_DATASETID
        - context.datasetId
        - modal.state.slyDatasetId
        - modal.state.inputDatasetId

    :param raise_not_found: if True, raises KeyError if dataset id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: dataset id
    :rtype: int
    """
    return _parse_from_env(
        name="dataset_id",
        keys=[
            "CONTEXT_DATASETID",
            "context.datasetId",
            "modal.state.slyDatasetId",
            "DATASET_ID",
            "modal.state.inputDatasetId",
        ],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def team_files_folder(raise_not_found: Optional[bool] = True) -> str:
    """Returns path to the team files folder from environment variable using following keys:
        - CONTEXT_SLYFOLDER
        - context.slyFolder
        - modal.state.slyFolder
        - FOLDER
    NOTE: same as folder
    :param raise_not_found: if True, raises KeyError if team files folder is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the team files folder
    :rtype: str
    """
    return _parse_from_env(
        name="team_files_folder",
        keys=[
            "CONTEXT_SLYFOLDER",
            "context.slyFolder",
            "modal.state.slyFolder",
            "FOLDER",
        ],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def folder(raise_not_found: Optional[bool] = True) -> str:
    """Returns path to the team files folder from environment variable using following keys:
        - CONTEXT_SLYFOLDER
        - context.slyFolder
        - modal.state.slyFolder
        - FOLDER
    NOTE: Same as team_files_folder
    :param raise_not_found: if True, raises KeyError if team files folder is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the team files folder
    :rtype: str
    """
    return team_files_folder(raise_not_found)


def team_files_file(raise_not_found: Optional[bool] = True) -> str:
    """Returns path to the file in the team files from environment variable using following keys:
        - CONTEXT_SLYFILE
        - context.slyFile
        - modal.state.slyFile
        - FILE

    NOTE: same as file
    :param raise_not_found: if True, raises KeyError if file is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the file in the team files
    :rtype: str
    """
    return _parse_from_env(
        name="team_files_file",
        keys=["CONTEXT_SLYFILE", "context.slyFile", "modal.state.slyFile", "FILE"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def server_address(raise_not_found: Optional[bool] = True) -> str:
    """Returns server address from environment variable using following keys:
        - SERVER_ADDRESS

    :param raise_not_found: if True, raises KeyError if server address is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: server address
    :rtype: str
    """
    return _parse_from_env(
        name="server_address",
        keys=["SERVER_ADDRESS"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def api_token(raise_not_found: Optional[bool] = True) -> str:
    """Returns an API token from environment variable using following keys:
        - API_TOKEN

    :param raise_not_found: if True, raises KeyError if API token is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: API token
    :rtype: str
    """
    return _parse_from_env(
        name="api_token",
        keys=["API_TOKEN"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def spawn_api_token(raise_not_found: Optional[bool] = True) -> str:
    """Returns SPAWN API token from environment variable using following keys:
        - context.spawnApiToken

    :param raise_not_found: if True, raises KeyError if API token is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: API token
    :rtype: str
    """
    return _parse_from_env(
        name="spawn_api_token",
        keys=["context.spawnApiToken"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def file(raise_not_found: Optional[bool] = True) -> str:
    """Returns path to the file in the team files from environment variable using following keys:
        - CONTEXT_SLYFILE
        - context.slyFile
        - modal.state.slyFile
        - FILE

    NOTE: Same as team_files_file
    :param raise_not_found: if True, raises KeyError if file is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the file in the team files
    :rtype: str
    """
    return team_files_file(raise_not_found)


def task_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns task id from environment variable using following keys:
        - TASK_ID

    :param raise_not_found: if True, raises KeyError if task id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: task id
    :rtype: int
    """
    return _parse_from_env(
        name="task_id",
        keys=["TASK_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def user_login(raise_not_found: Optional[bool] = True) -> str:
    """Returns user login from environment variable using following keys:
        - USER_LOGIN
        - CONTEXT_USERLOGIN
        - context.userLogin

    :param raise_not_found: if True, raises KeyError if user login is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: user login
    :rtype: str
    """
    return _parse_from_env(
        name="user_login",
        keys=["USER_LOGIN", "context.userLogin", "CONTEXT_USERLOGIN"],
        postprocess_fn=lambda x: str(x),
        default="user (debug)",
        raise_not_found=raise_not_found,
    )


def app_name(raise_not_found: Optional[bool] = True) -> str:
    """Returns application's name from environment variable using following keys:
        - APP_NAME

    :param raise_not_found: if True, raises KeyError if application's name is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: application's name
    :rtype: str
    """
    return _parse_from_env(
        name="app_name",
        keys=["APP_NAME"],
        postprocess_fn=lambda x: str(x),
        default="Supervisely App (debug)",
        raise_not_found=raise_not_found,
    )


def user_id(raise_not_found: Optional[bool] = True) -> int:
    """Returns user id from environment variable using following keys:
        - USER_ID
        - CONTEXT_USERID
        - context.userId

    :param raise_not_found: if True, raises KeyError if user id is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: user id
    :rtype: int
    """
    return _parse_from_env(
        name="user_id",
        keys=["USER_ID", "context.userId", "CONTEXT_USERID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def content_origin_update_interval() -> float:
    """Returns interval of updating the content origin from environment variable using following keys:
        - CONTENT_ORIGIN_UPDATE_INTERVAL

    :return: content origin update interval
    :rtype: float
    """
    return _parse_from_env(
        name="content_origin_update_interval",
        keys=["CONTENT_ORIGIN_UPDATE_INTERVAL"],
        postprocess_fn=lambda x: float(x),
        default=0.5,
        raise_not_found=False,
    )


def smart_cache_ttl(
    raise_not_found: Optional[bool] = False, default: Optional[int] = 30 * 60
) -> int:
    """Returns TTL of the smart cache from environment variable using following keys:
        - SMART_CACHE_TTL

    :param raise_not_found: if True, raises KeyError if smart cache TTL is not found in environment variables
    :type raise_not_found: Optional[bool]
    :param default: default value of smart cache TTL
    :type default: Optional[int]
    :return: smart cache TTL
    :rtype: int
    """
    return _parse_from_env(
        name="smart_cache_ttl",
        keys=["SMART_CACHE_TTL"],
        postprocess_fn=lambda x: max(int(x), 1),
        default=default,
        raise_not_found=raise_not_found,
    )


def smart_cache_size(raise_not_found: Optional[bool] = False, default: Optional[int] = 256) -> int:
    """Returns the size of the smart cache from environment variable using following keys:
        - SMART_CACHE_SIZE

    :param raise_not_found: if True, raises KeyError if smart cache size is not found in environment variables
    :type raise_not_found: Optional[bool]
    :default: default value of smart cache size
    :type default: Optional[int]
    :return: smart cache size
    :rtype: int
    """
    return _parse_from_env(
        name="smart_cache_size",
        keys=["SMART_CACHE_SIZE"],
        postprocess_fn=lambda x: max(int(x), 1),
        default=default,
        raise_not_found=raise_not_found,
    )


def smart_cache_container_dir(default: Optional[str] = "/tmp/smart_cache") -> str:
    """Returns a path to the smart cache dir in the container from environment variable using following keys:
        - SMART_CACHE_CONTAINER_DIR

    :param default: default value of smart cache container dir
    :type default: Optional[str]
    :return: path to the smart cache dir in the container
    :rtype: str
    """
    return _parse_from_env(
        name="smart_cache_container_dir",
        keys=["SMART_CACHE_CONTAINER_DIR"],
        default=default,
        raise_not_found=False,
        postprocess_fn=lambda x: x.strip(),
    )


def autostart() -> bool:
    """Returns autostart flag from environment variable using following keys:
        - modal.state.autostart

    :return: autostart flag
    :rtype: bool
    """
    return _parse_from_env(
        name="autostart",
        keys=["modal.state.autostart", "AUTOSTART"],
        default=False,
        raise_not_found=False,
        postprocess_fn=flag_from_env,
    )


def set_autostart(value: Optional[Literal["1", "true", "yes"]] = None) -> None:
    """Set modal.state.autostart env to the given value.
    Possible values to set the autostart to True:
        - "1"
        - "true"
        - "yes"

    To remove the variable, use `value=None`, or omit the argument.

    :param value: value to set the autostart to
    :type value: Optional[Union[Literal["1", "true", "yes"]], None]
    """
    if value is None:
        os.environ.pop("modal.state.autostart", None)
        return

    if not flag_from_env(value):
        raise ValueError("Unknown value for `autostart` env. Use `1`, `true`, `yes` or None.")
    os.environ["modal.state.autostart"] = value


def apps_cache_dir():
    """Returns apps cache directory path from environment variable using following keys:
        - APPS_CACHE_DIR

    :return: apps cache directory path
    :rtype: str
    """
    return _parse_from_env(
        name="apps_cache_dir",
        keys=["APPS_CACHE_DIR"],
        postprocess_fn=lambda x: x,
        default="/apps_cache",
        raise_not_found=False,
    )


def mininum_instance_version_for_sdk() -> str:
    """Returns minimum instance version required by the SDK from environment variable using following
        - MINIMUM_INSTANCE_VERSION_FOR_SDK

    :return: minimum instance version required by the SDK
    :rtype: str
    """
    return _parse_from_env(
        name="sdk_minimum_instance_version",
        keys=["MINIMUM_INSTANCE_VERSION_FOR_SDK"],
        postprocess_fn=lambda x: x,
        raise_not_found=False,
    )
