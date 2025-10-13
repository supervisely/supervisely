# coding: utf-8
import json
import os
from contextvars import ContextVar, Token
from typing import Callable, List, Literal, Optional, Union

RAISE_IF_NOT_FOUND = True
_MULTIUSER_USER_CTX: ContextVar[Optional[Union[int, str]]] = ContextVar(
    "supervisely_multiuser_app_user_id",
    default=None,
)

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


def _parse_list_from_env(value: str) -> List[str]:
    import ast

    return [str(x).strip() for x in ast.literal_eval(value)]


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
            f"{name} is not defined as environment variable. One of the envs has to be defined: {keys}. Learn more in developer portal: https://developer.supervisely.com/getting-started/environment-variables"
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


def team_files_folders(raise_not_found: Optional[bool] = True) -> List[str]:
    """Returns paths to the team files folders from environment variable using following keys:
        - CONTEXT_SLYFOLDERS
        - context.slyFolders
        - modal.state.slyFolders
        - FOLDERS
    NOTE: same as team_files_folders
    :param raise_not_found: if True, raises KeyError if team files folders are not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the team files folders
    :rtype: str
    """
    return _parse_from_env(
        name="team_files_folders",
        keys=[
            "CONTEXT_SLYFOLDERS",
            "context.slyFolders",
            "modal.state.slyFolders",
            "FOLDERS",
        ],
        postprocess_fn=_parse_list_from_env,
        default=[],
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


def folders(raise_not_found: Optional[bool] = True) -> List[str]:
    """Returns paths to the team files folders from environment variable using following keys:
        - CONTEXT_SLYFOLDERS
        - context.slyFolders
        - modal.state.slyFolders
        - FOLDERS
    NOTE: Same as team_files_folders
    :param raise_not_found: if True, raises KeyError if team files folders are not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the team files folders
    :rtype: str
    """
    return team_files_folders(raise_not_found)


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


def team_files_files(raise_not_found: Optional[bool] = True) -> List[str]:
    """Returns paths to the files in the team files from environment variable using following keys:
        - CONTEXT_SLYFILES
        - context.slyFiles
        - modal.state.slyFiles
        - FILES

    NOTE: same as team_files_file
    :param raise_not_found: if True, raises KeyError if file is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the file in the team files
    :rtype: str
    """
    return _parse_from_env(
        name="team_files_files",
        keys=["CONTEXT_SLYFILES", "context.slyFiles", "modal.state.slyFiles", "FILES"],
        postprocess_fn=_parse_list_from_env,
        default=[],
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


def files(raise_not_found: Optional[bool] = True) -> List[str]:
    """Returns paths to the files in the team files from environment variable using following keys:
        - CONTEXT_SLYFILES
        - context.slyFiles
        - modal.state.slyFiles
        - FILES

    NOTE: Same as team_files_files
    :param raise_not_found: if True, raises KeyError if file is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: path to the file in the team files
    :rtype: str
    """
    return team_files_files(raise_not_found)


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


def semaphore_size() -> int:
    """Returns semaphore size from environment variable using following
        - SUPERVISELY_ASYNC_SEMAPHORE

    :return: semaphore size
    :rtype: int
    """
    return _parse_from_env(
        name="semaphore_size",
        keys=["SUPERVISELY_ASYNC_SEMAPHORE"],
        postprocess_fn=lambda x: int(x),
        raise_not_found=False,
    )


def supervisely_server_path_prefix() -> str:
    """Returns routes prefix from environment variable using following
        - SUPERVISELY_SERVER_PATH_PREFIX

    :return: routes prefix
    :rtype: str
    """
    return _parse_from_env(
        name="supervisely_server_path_prefix",
        keys=["SUPERVISELY_SERVER_PATH_PREFIX"],
        postprocess_fn=lambda x: x,
        default="",
        raise_not_found=False,
    )


def supervisely_skip_https_user_helper_check() -> bool:
    """Returns decision to skip `_check_https_redirect` for API from environment variable using following
        - SUPERVISELY_SKIP_HTTPS_USER_HELPER_CHECK"

    :return: decision to skip `_check_https_redirect` for API
    :rtype: bool
    """
    return _parse_from_env(
        name="supervisely_skip_https_user_helper_check",
        keys=["SUPERVISELY_SKIP_HTTPS_USER_HELPER_CHECK"],
        postprocess_fn=flag_from_env,
        default=False,
        raise_not_found=False,
    )


def configure_minimum_instance_version() -> None:
    """
    Configure MINIMUM_INSTANCE_VERSION_FOR_SDK environment variable
    from the latest entry in versions.json file.

    This function should be called during SDK initialization to automatically
    set the minimum required instance version based on the versions.json file.
    """
    from supervisely._utils import get_latest_instance_version_from_json

    latest_version = get_latest_instance_version_from_json()
    if latest_version:
        os.environ["MINIMUM_INSTANCE_VERSION_FOR_SDK"] = latest_version

def app_categories(raise_not_found: Optional[bool] = False) -> list:
    """Returns a list of app categories from environment variable using following keys:
        - APP_CATEGORIES
    :param raise_not_found: if True, raises KeyError if app category is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: app categories
    :rtype: list
    """
    return _parse_from_env(
        name="app_category",
        keys=["APP_CATEGORIES"],
        postprocess_fn=lambda x: json.loads(x),
        default=[],
        raise_not_found=raise_not_found,
    )


def upload_count(raise_not_found: Optional[bool] = False) -> dict:
    """Returns a dictionary of upload counts from environment variable using following
        - UPLOAD_COUNT
    :param raise_not_found: if True, raises KeyError if upload count is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: upload count
    :rtype: dict
    """
    return _parse_from_env(
        name="upload_count",
        keys=["UPLOAD_COUNT"],
        postprocess_fn=lambda x: json.loads(x),
        default={},
        raise_not_found=raise_not_found,
    )


def uploaded_ids(raise_not_found: Optional[bool] = False) -> dict:
    """Returns a dictionary with dataset IDs as keys and lists of uploaded IDs as values from environment variable using following
        - UPLOADED_IDS
    :param raise_not_found: if True, raises KeyError if uploaded IDs is not found in environment variables
    :type raise_not_found: Optional[bool]
    :return: uploaded IDs
    :rtype: dict
    """
    return _parse_from_env(
        name="uploaded_ids",
        keys=["UPLOADED_IDS"],
        postprocess_fn=lambda x: json.loads(x),
        default={},
        raise_not_found=raise_not_found,
    )


def increment_upload_count(dataset_id: int, count: int = 1) -> None:
    """Increments the upload count for the given dataset id by the specified count.

    :param dataset_id: The dataset id to increment the upload count for.
    :type dataset_id: int
    :param count: The amount to increment the upload count by. Defaults to 1.
    :type count: int
    """
    upload_info = upload_count()
    upload_info[str(dataset_id)] = upload_info.get(str(dataset_id), 0) + count
    os.environ["UPLOAD_COUNT"] = json.dumps(upload_info)


def add_uploaded_ids_to_env(dataset_id: int, ids: List[int]) -> None:
    """Adds the list of uploaded IDs to the environment variable for the given dataset ID.

    :param dataset_id: The dataset ID to associate the uploaded IDs with.
    :type dataset_id: int
    :param ids: The list of uploaded IDs to add.
    :type ids: List[int]
    """
    uploaded = uploaded_ids()
    if str(dataset_id) not in uploaded:
        uploaded[str(dataset_id)] = []
    existing_ids = set(uploaded[str(dataset_id)])
    if set(ids).intersection(existing_ids):
        for _id in ids:
            if _id not in existing_ids:
                uploaded[str(dataset_id)].append(_id)
    else:
        uploaded[str(dataset_id)].extend(ids)
    os.environ["UPLOADED_IDS"] = json.dumps(uploaded)


def is_multiuser_mode_enabled() -> bool:
    """Returns multiuser app mode flag from environment variable using following keys:
        - SUPERVISELY_MULTIUSER_APP_MODE
    :return: multiuser app mode flag
    :rtype: bool
    """
    return _parse_from_env(
        name="is_multiuser_mode_enabled",
        keys=["SUPERVISELY_MULTIUSER_APP_MODE"],
        default=False,
        raise_not_found=False,
        postprocess_fn=flag_from_env,
    )


def enable_multiuser_app_mode() -> None:
    """
    Enables multiuser app mode by setting the environment variable.
    This function can be used to activate multiuser mode in the application allowing
    separation of user DataJson/StateJson.
    """
    os.environ["SUPERVISELY_MULTIUSER_APP_MODE"] = "true"


def disable_multiuser_app_mode() -> None:
    """Disables multiuser app mode by removing the environment variable."""
    os.environ.pop("SUPERVISELY_MULTIUSER_APP_MODE", None)


def set_user_for_multiuser_app(user_id: Optional[Union[int, str]]) -> Token:
    """
    Sets the user ID for multiuser app mode by setting the environment variable.
    This function should be used in multiuser mode to separate user DataJson/StateJson.

    :param user_id: The user ID (or session key) to set for the current request.
    :type user_id: int | str
    :return: A context token that can be used to reset the user ID later.
    :rtype: Token
    :raises RuntimeError: If multiuser app mode is not enabled.
    """
    if not is_multiuser_mode_enabled():
        raise RuntimeError("Multiuser app mode is not enabled. Cannot set user ID.")
    return _MULTIUSER_USER_CTX.set(user_id)


def reset_user_for_multiuser_app(token: Token) -> None:
    """
    Resets the user ID for multiuser app mode using the provided context token.

    :param token: Context token obtained from `set_user_for_multiuser_app`.
    :type token: Token
    """
    if not is_multiuser_mode_enabled():
        return
    _MULTIUSER_USER_CTX.reset(token)


def user_from_multiuser_app() -> Optional[Union[int, str]]:
    """
    Retrieves the user ID for multiuser app mode from the environment variable.

    :return: The user ID if set, otherwise None.
    :rtype: Optional[Union[int, str]]
    """
    if not is_multiuser_mode_enabled():
        return None
    user_id = _MULTIUSER_USER_CTX.get(None)
    if user_id is not None:
        return user_id
