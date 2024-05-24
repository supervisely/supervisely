# coding: utf-8
"""api connection to the server which allows user to communicate with Supervisely"""

from __future__ import annotations

import datetime
import gc
import glob
import json
import os
import shutil
from logging import Logger
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urljoin, urlparse

import jwt
import requests
from dotenv import get_key, load_dotenv, set_key
from pkg_resources import parse_version
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

import supervisely.api.advanced_api as advanced_api
import supervisely.api.agent_api as agent_api
import supervisely.api.annotation_api as annotation_api
import supervisely.api.app_api as app_api
import supervisely.api.dataset_api as dataset_api
import supervisely.api.file_api as file_api
import supervisely.api.github_api as github_api
import supervisely.api.image_annotation_tool_api as image_annotation_tool_api
import supervisely.api.image_api as image_api
import supervisely.api.import_storage_api as import_stoarge_api
import supervisely.api.labeling_job_api as labeling_job_api
import supervisely.api.neural_network_api as neural_network_api
import supervisely.api.object_class_api as object_class_api
import supervisely.api.plugin_api as plugin_api
import supervisely.api.pointcloud.pointcloud_api as pointcloud_api
import supervisely.api.pointcloud.pointcloud_episode_api as pointcloud_episode_api
import supervisely.api.project_api as project_api
import supervisely.api.remote_storage_api as remote_storage_api
import supervisely.api.report_api as report_api
import supervisely.api.role_api as role_api
import supervisely.api.storage_api as storage_api
import supervisely.api.task_api as task_api
import supervisely.api.team_api as team_api
import supervisely.api.user_api as user_api
import supervisely.api.video.video_api as video_api
import supervisely.api.video_annotation_tool_api as video_annotation_tool_api
import supervisely.api.volume.volume_api as volume_api
import supervisely.api.workspace_api as workspace_api
import supervisely.io.env as sly_env
from supervisely._utils import camel_to_snake, is_community, is_development
from supervisely.api.module_api import ApiField
from supervisely.io.network_exceptions import (
    process_requests_exception,
    process_unhandled_request,
)
from supervisely.sly_logger import logger
from supervisely.project.project_meta import ProjectMeta

SUPERVISELY_TASK_ID = "SUPERVISELY_TASK_ID"
SUPERVISELY_PUBLIC_API_RETRIES = "SUPERVISELY_PUBLIC_API_RETRIES"
SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC = "SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC"
SERVER_ADDRESS = "SERVER_ADDRESS"
SUPERVISELY_API_SERVER_ADDRESS = "SUPERVISELY_API_SERVER_ADDRESS"
API_TOKEN = "API_TOKEN"
TASK_ID = "TASK_ID"
SUPERVISELY_ENV_FILE = os.path.join(Path.home(), "supervisely.env")


class ApiContext:
    """
    Context manager for the API object for optimization purposes.
    Use this context manager when you need to perform a series of operations on the same project or dataset.
    It allows you to avoid redundant API calls to get the same project or dataset info multiple times.

    :param api: API object.
    :type api: :class:`Api`
    :param project_id: Project ID.
    :type project_id: int, optional
    :param dataset_id: Dataset ID.
    :type dataset_id: int, optional
    :param project_meta: ProjectMeta object.
    :type project_meta: :class:`ProjectMeta`, optional
    :raises: :class:`RuntimeError`, if api is None.

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

            with ApiContext(
                api,
                project_id=33333,
                dataset_id=99999,
                project_meta=project_meta,
                with_alpha_masks=True,
            ):
                api.annotation.upload_paths(image_ids, ann_paths, anns_progress)
                # another code here
    """

    def __init__(
        self,
        api: Api,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        project_meta: Optional[ProjectMeta] = None,
        with_alpha_masks: Optional[bool] = True,
    ):
        if api is None:
            raise RuntimeError("Api object is None")
        self.api = api
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.project_meta = project_meta
        self.with_alpha_masks = with_alpha_masks

    def __enter__(self):
        self.api.optimization_context = {
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "project_meta": self.project_meta,
            "with_alpha_masks": self.with_alpha_masks,
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.optimization_context = {}


class UserSession:
    """
    UserSession object contains info that is returned after user authentication.

    :param server: Server url.
    :type server: str
    :raises: :class:`RuntimeError`, if server url is invalid.
    """

    def __init__(self, server_address: str):
        self.api_token = None
        self.team_id = None
        self.workspace_id = None
        self.server_address = server_address

        if not self._normalize_and_validate_server_url():
            raise RuntimeError(f"Invalid server url: {server_address}")

    def __str__(self):
        return f"UserSession(server={self.server_address})"

    def __repr__(self):
        return self.__str__()

    def _normalize_and_validate_server_url(self) -> bool:
        """
        Validate server url.

        :return: True if server url is valid, False otherwise.
        """
        self.server_address = Api.normalize_server_address(self.server_address)
        if not self.server_address.startswith("https://"):
            response = requests.get(self.server_address, allow_redirects=False)
            if (300 <= response.status_code < 400) or (
                response.headers.get("Location", "").startswith("https://")
            ):
                self.server_address = self.server_address.replace("http://", "https://")
        result = urlparse(self.server_address)
        if all([result.scheme, result.netloc]):
            try:
                response = requests.get(self.server_address)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
        return False

    def _setattrs_user_session(self, decoded_token):
        """
        Add decoded info to UserSession object.

        :param decoded_token: Decoded token.
        :type decoded_token: dict
        :return: None
        :rtype: :class:`NoneType`
        """
        for key, value in decoded_token.items():
            if key == "group":
                self.team = value
                self.team_id = value["id"]
            elif key == "workspace":
                self.workspace = value
                self.workspace_id = value["id"]
            else:
                key = camel_to_snake(key)
                setattr(self, key, value)

    def log_in(self, login: str, password: str) -> UserSession:
        """
        Authenticate user and return UserSession object with decoded info from JWT token.

        :param login: User login.
        :type login: str
        :param password: User password.
        :type password: str
        :return: UserSession object
        :rtype: :class:`UserSession`
        """
        login_url = urljoin(self.server_address, "api/account")
        payload = {"login": login, "password": password}
        response = requests.post(login_url, data=payload)
        del password
        gc.collect()
        if response.status_code == 200:
            data = response.json()
            jwt_token = data.get("token", None)
            decoded_token = jwt.decode(jwt_token, options={"verify_signature": False})
            self._setattrs_user_session(decoded_token)
            return self
        else:
            raise RuntimeError(f"Failed to authenticate user: status code {response.status_code}")


class Api:
    """
    An API connection to the server with which you can communicate with your teams, workspaces and projects. :class:`Api<Api>` object is immutable.

    :param server_address: Address of the server.
    :type server_address: str
    :param token: Unique secret token associated with your agent.
    :type token: str
    :param retry_count: The number of attempts to connect to the server.
    :type retry_count: int, optional
    :param retry_sleep_sec: The number of seconds to delay between attempts to connect to the server.
    :type retry_sleep_sec: int, optional
    :param external_logger: Logger class object.
    :type external_logger: logger, optional
    :param ignore_task_id:
    :type ignore_task_id: bool, optional
    :param api_server_address: Address of the API server.
    :type api_server_address: str, optional
    :param check_instance_version: Check if the given version is lower or equal to the current
        Supervisely instance version. If set to True, will try to read the version from the environment variable
        "MINIMUM_INSTANCE_VERSION_FOR_SDK". If set to a string, will use this string as the version to check.
        If set to False, will skip the check.
    :type check_instance_version: bool or str, optional
    :raises: :class:`ValueError`, if token is None or it length != 128
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
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")
    """

    def __init__(
        self,
        server_address: str = None,
        token: str = None,
        retry_count: Optional[int] = 10,
        retry_sleep_sec: Optional[int] = None,
        external_logger: Optional[Logger] = None,
        ignore_task_id: Optional[bool] = False,
        api_server_address: str = None,
        check_instance_version: Union[bool, str] = False,
    ):
        if server_address is None and token is None:
            server_address = os.environ.get(SERVER_ADDRESS, None)
            token = os.environ.get(API_TOKEN, None)

        if server_address is None:
            raise ValueError(
                "SERVER_ADDRESS env variable is undefined, https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        if token is None:
            raise ValueError(
                "API_TOKEN env variable is undefined, https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        self.server_address = Api.normalize_server_address(server_address)

        self._api_server_address = None
        if api_server_address is None:
            api_server_address = os.environ.get(SUPERVISELY_API_SERVER_ADDRESS, None)

        if api_server_address is not None:
            self._api_server_address = Api.normalize_server_address(api_server_address)

        if retry_count is None:
            retry_count = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRIES, "10"))
        if retry_sleep_sec is None:
            retry_sleep_sec = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC, "1"))

        if len(token) != 128:
            raise ValueError("Invalid token {!r}: length != 128".format(token))

        self.token = token
        self.headers = {"x-api-key": token}
        self.task_id = os.getenv(SUPERVISELY_TASK_ID)
        if self.task_id is not None and ignore_task_id is False:
            self.headers["x-task-id"] = self.task_id
        self.context = {}
        self.additional_fields = {}
        self.optimization_context = {}

        self.team = team_api.TeamApi(self)
        self.workspace = workspace_api.WorkspaceApi(self)
        self.project = project_api.ProjectApi(self)
        self.model = neural_network_api.NeuralNetworkApi(self)
        self.task = task_api.TaskApi(self)
        self.dataset = dataset_api.DatasetApi(self)
        self.image = image_api.ImageApi(self)
        self.annotation = annotation_api.AnnotationApi(self)
        self.plugin = plugin_api.PluginApi(self)
        self.agent = agent_api.AgentApi(self)
        self.role = role_api.RoleApi(self)
        self.user = user_api.UserApi(self)
        self.labeling_job = labeling_job_api.LabelingJobApi(self)
        self.video = video_api.VideoApi(self)
        # self.project_class = project_class_api.ProjectClassApi(self)
        self.object_class = object_class_api.ObjectClassApi(self)
        self.report = report_api.ReportApi(self)
        self.pointcloud = pointcloud_api.PointcloudApi(self)
        self.pointcloud_episode = pointcloud_episode_api.PointcloudEpisodeApi(self)
        self.app = app_api.AppApi(self)
        self.file = file_api.FileApi(self)
        self.storage = storage_api.StorageApi(self)
        self.img_ann_tool = image_annotation_tool_api.ImageAnnotationToolApi(self)
        self.vid_ann_tool = video_annotation_tool_api.VideoAnnotationToolApi(self)
        self.advanced = advanced_api.AdvancedApi(self)
        self.import_storage = import_stoarge_api.ImportStorageApi(self)
        self.remote_storage = remote_storage_api.RemoteStorageApi(self)
        self.github = github_api.GithubApi(self)
        self.volume = volume_api.VolumeApi(self)

        self.retry_count = retry_count
        self.retry_sleep_sec = retry_sleep_sec

        self.logger = external_logger or logger

        self._require_https_redirect_check = not self.server_address.startswith("https://")

        if check_instance_version:
            self._check_version(None if check_instance_version is True else check_instance_version)

    @classmethod
    def normalize_server_address(cls, server_address: str) -> str:
        """ """
        result = server_address.strip("/")
        if ("http://" not in result) and ("https://" not in result):
            result = "http://" + result
        return result

    @classmethod
    def from_env(
        cls,
        retry_count: int = 10,
        ignore_task_id: bool = False,
        env_file: str = SUPERVISELY_ENV_FILE,
        check_instance_version: Union[bool, str] = False,
    ) -> Api:
        """
        Initialize API use environment variables.

        :param retry_count: The number of attempts to connect to the server.
        :type retry_count: int
        :param ignore_task_id:
        :type ignore_task_id: bool
        :param env_file: Path to your .env file.
        :type env_file: str
        :param check_instance_version: Check if the given version is lower or equal to the current
            version of the Supervisely instance.
        :type check_instance_version: bool or str, optional
        :return: Api object
        :rtype: :class:`Api<supervisely.api.api.Api>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            api = sly.Api.from_env()

            # alternatively you can store SERVER_ADDRESS and API_TOKEN
            # in "~/supervisely.env" .env file
            # Learn more here: https://developer.supervise.ly/app-development/basics/add-private-app#create-.env-file-supervisely.env-with-the-following-content-learn-more-here

            api = sly.Api.from_env()
        """

        server_address = sly_env.server_address(raise_not_found=False)
        token = sly_env.api_token(raise_not_found=False)

        if is_development() and None in (server_address, token):
            env_path = os.path.expanduser(env_file)
            if os.path.exists(env_path):
                _, extension = os.path.splitext(env_path)
                if extension == ".env":
                    load_dotenv(env_path)
                    server_address = sly_env.server_address()
                    token = sly_env.api_token()
                else:
                    raise ValueError(f"'{env_path}' is not an '*.env' file")
            else:
                raise FileNotFoundError(f"File not found: '{env_path}'")

        if server_address is None:
            raise ValueError(
                "SERVER_ADDRESS env variable is undefined. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication"
            )
        if token is None:
            raise ValueError(
                "API_TOKEN env variable is undefined. Learn more here: https://developer.supervise.ly/getting-started/basics-of-authentication"
            )

        return cls(
            server_address,
            token,
            retry_count=retry_count,
            ignore_task_id=ignore_task_id,
            check_instance_version=check_instance_version,
        )

    def add_header(self, key: str, value: str) -> None:
        """
        Add given key and value to headers dictionary.

        :param key: New key.
        :type key: str
        :param value: New value.
        :type value: str
        :raises: :class:`RuntimeError`, if key is already set
        :return: None
        :rtype: :class:`NoneType`
        """
        if key in self.headers:
            raise RuntimeError(
                f"Header {key!r} is already set for the API object. "
                f"Current value: {self.headers[key]!r}. Tried to set value: {value!r}"
            )
        self.headers[key] = value

    def add_additional_field(self, key: str, value: str) -> None:
        """
        Add given key and value to additional_fields dictionary.

        :param key: New key.
        :type key: str
        :param value: New value.
        :type value: str
        :return: None
        :rtype: :class:`NoneType`
        """
        self.additional_fields[key] = value

    @property
    def instance_version(self) -> str:
        """Return Supervisely instance version, e.g. "6.9.13".
        If the version cannot be determined, return "unknown".

        :return: Supervisely instance version or "unknown" if the version cannot be determined.
        :rtype: str

        :Usage example:

        .. code-block:: python

                import supervisely as sly

                api = sly.Api(server_address='https://app.supervisely.com', token='4r47N...xaTatb')
                print(api.instance_version)
                # Output:
                # '6.9.13'
        """
        try:
            version = self.post("instance.version", {}).json().get(ApiField.VERSION)
        except Exception as e:
            logger.warning(f"Failed to get instance version from server: {e}")
            version = "unknown"
        return version

    def is_version_supported(self, version: Optional[str] = None) -> Union[bool, None]:
        """Check if the given version is lower or equal to the current Supervisely instance version.
        If the version omitted, will try to read it from the environment variable "MINIMUM_INSTANCE_VERSION_FOR_SDK".
        If the version is lower or equal, return True, otherwise False.
        If the version of the instance cannot be determined, return False.

        :param version: Version to check.
        :type version: Optional[str], e.g. "6.9.13"
        :return: True if the given version is lower or equal to the current Supervisely
            instance version, otherwise False.
        :rtype: bool

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api(server_address='https://app.supervisely.com', token='4r47N...xaTatb')
            version_to_check = "6.9.13"
            print(api.is_version_supported(version_to_check))
            # Output:
            # True
        """
        instance_version = self.instance_version
        if instance_version == "unknown":
            return

        if not version:
            version = sly_env.mininum_instance_version_for_sdk()
            if not version:
                logger.debug(
                    "Cant find MINIMUM_INSTANCE_VERSION_FOR_SDK in environment variables, "
                    "check of the minimum version is skipped."
                )
                return

        try:
            version = str(version)
        except Exception:
            logger.warning(
                f"Provided version {version!r} is not a valid version string "
                f"(expected format: 'x.y.z'). The output of this function will be incorrect."
            )
            return

        return parse_version(instance_version) >= parse_version(version)

    def _check_version(self, version: Optional[str] = None) -> None:
        """Check if the given version is compatible with the current Supervisely instance version.
        Compatible means that the given version is lower or equal to the current Supervisely instance version.
        If check was not successful, log a debug message, if the version is not supported, log a warning message.

        :param version: Version to check.
        :type version: Optional[str], e.g. "6.9.13"
        """

        # Since it's a informational message, we don't raise an exception if the check fails
        # in any case, we don't want to interrupt the user's workflow.
        try:
            check_result = self.is_version_supported(version)
            if check_result is None:
                logger.debug(
                    "Failed to check if the instance version meets the minimum requirements "
                    "of current SDK version. "
                    "Ensure that the MINIMUM_INSTANCE_VERSION_FOR_SDK environment variable is set. "
                    "Usually you can ignore this message, but if you're adding new features, "
                    "which will require upgrade of the Supervisely instance, you should update "
                    "it supervisely.__init__.py file."
                )
            if check_result is False:
                message = (
                    "The current version of the Supervisely instance is not supported by the SDK. "
                    "Some features may not work correctly."
                )
                if not is_community():
                    message += (
                        " Please upgrade the Supervisely instance to the latest version (recommended) "
                        "or downgrade the SDK to the version that supports the current instance (not recommended). "
                        "Refer to this docs for more information: "
                        "https://docs.supervisely.com/enterprise-edition/get-supervisely/upgrade "
                        "Check out changelog for the latest version of Supervisely: "
                        "https://app.supervisely.com/changelog"
                    )
                    logger.warning(message)
        except Exception as e:
            logger.debug(
                f"Tried to check version compatibility between SDK and instance, but failed: {e}"
            )

    def post(
        self,
        method: str,
        data: Dict,
        retries: Optional[int] = None,
        stream: Optional[bool] = False,
        raise_error: Optional[bool] = False,
    ) -> requests.Response:
        """
        Performs POST request to server with given parameters.

        :param method: Method name.
        :type method: str
        :param data: Dictionary to send in the body of the :class:`Request`.
        :type data: dict
        :param retries: The number of attempts to connect to the server.
        :type retries: int, optional
        :param stream: Define, if you'd like to get the raw socket response from the server.
        :type stream: bool, optional
        :param raise_error: Define, if you'd like to raise error if connection is failed. Retries will be ignored.
        :type raise_error: bool, optional
        :return: Response object
        :rtype: :class:`Response<Response>`
        """
        self._check_https_redirect()
        if retries is None:
            retries = self.retry_count

        url = self.api_server_address + "/v3/" + method
        logger.trace(f"POST {url}")

        for retry_idx in range(retries):
            response = None
            try:
                if type(data) is bytes:
                    response = requests.post(url, data=data, headers=self.headers, stream=stream)
                elif type(data) is MultipartEncoderMonitor or type(data) is MultipartEncoder:
                    response = requests.post(
                        url,
                        data=data,
                        headers={**self.headers, "Content-Type": data.content_type},
                        stream=stream,
                    )
                else:
                    json_body = data
                    if type(data) is dict:
                        json_body = {**data, **self.additional_fields}
                    response = requests.post(
                        url, json=json_body, headers=self.headers, stream=stream
                    )

                if response.status_code != requests.codes.ok:  # pylint: disable=no-member
                    self._check_version()
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                if raise_error:
                    raise exc
                else:
                    process_requests_exception(
                        self.logger,
                        exc,
                        method,
                        url,
                        verbose=True,
                        swallow_exc=True,
                        sleep_sec=min(self.retry_sleep_sec * (2**retry_idx), 60),
                        response=response,
                        retry_info={"retry_idx": retry_idx + 1, "retry_limit": retries},
                    )
            except Exception as exc:
                process_unhandled_request(self.logger, exc)
        raise requests.exceptions.RetryError("Retry limit exceeded ({!r})".format(url))

    def get(
        self,
        method: str,
        params: Dict,
        retries: Optional[int] = None,
        stream: Optional[bool] = False,
        use_public_api: Optional[bool] = True,
    ) -> requests.Response:
        """
        Performs GET request to server with given parameters.

        :param method:
        :type method: str
        :param params: Dictionary to send in the body of the :class:`Request`.
        :type method: dict
        :param retries: The number of attempts to connect to the server.
        :type method: int, optional
        :param stream: Define, if you'd like to get the raw socket response from the server.
        :type method: bool, optional
        :param use_public_api:
        :type method: bool, optional
        :return: Response object
        :rtype: :class:`Response<Response>`
        """
        self._check_https_redirect()
        if retries is None:
            retries = self.retry_count

        url = self.api_server_address + "/v3/" + method
        if use_public_api is False:
            url = os.path.join(self.server_address, method)
        logger.trace(f"GET {url}")

        for retry_idx in range(retries):
            response = None
            try:
                json_body = params
                if type(params) is dict:
                    json_body = {**params, **self.additional_fields}
                response = requests.get(url, params=json_body, headers=self.headers, stream=stream)

                if response.status_code != requests.codes.ok:  # pylint: disable=no-member
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(
                    self.logger,
                    exc,
                    method,
                    url,
                    verbose=True,
                    swallow_exc=True,
                    sleep_sec=min(self.retry_sleep_sec * (2**retry_idx), 60),
                    response=response,
                    retry_info={"retry_idx": retry_idx + 2, "retry_limit": retries},
                )
            except Exception as exc:
                process_unhandled_request(self.logger, exc)

    @staticmethod
    def _raise_for_status(response):
        """
        Raise error and show message with error code if given response can not connect to server.
        :param response: Request class object
        """
        http_error_msg = ""
        if isinstance(response.reason, bytes):
            try:
                reason = response.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = response.reason.decode("iso-8859-1")
        else:
            reason = response.reason

        if 400 <= response.status_code < 500:
            http_error_msg = "%s Client Error: %s for url: %s (%s)" % (
                response.status_code,
                reason,
                response.url,
                response.content.decode("utf-8"),
            )

        elif 500 <= response.status_code < 600:
            http_error_msg = "%s Server Error: %s for url: %s (%s)" % (
                response.status_code,
                reason,
                response.url,
                response.content.decode("utf-8"),
            )

        if http_error_msg:
            raise requests.exceptions.HTTPError(http_error_msg, response=response)

    @staticmethod
    def parse_error(
        response: requests.Response,
        default_error: Optional[str] = "Error",
        default_message: Optional[str] = "please, contact administrator",
    ):
        """
        Processes error from response.

        :param response: Request object.
        :type method: Request
        :param default_error: Error description.
        :type method: str, optional
        :param default_message: Message to user.
        :type method: str, optional
        :return: Number of error and message about curren connection mistake
        :rtype: :class:`int`, :class:`str`
        """
        ERROR_FIELD = "error"
        MESSAGE_FIELD = "message"
        DETAILS_FIELD = "details"

        try:
            data_str = response.content.decode("utf-8")
            data = json.loads(data_str)
            error = data.get(ERROR_FIELD, default_error)
            details = data.get(DETAILS_FIELD, {})
            if type(details) is dict:
                message = details.get(MESSAGE_FIELD, default_message)
            else:
                message = details[0].get(MESSAGE_FIELD, default_message)

            return error, message
        except Exception:
            return "", ""

    def pop_header(self, key: str) -> str:
        """ """
        if key not in self.headers:
            raise KeyError(f"Header {key!r} not found")
        return self.headers.pop(key)

    def _check_https_redirect(self):
        if self._require_https_redirect_check is True:
            response = requests.get(self.server_address, allow_redirects=False)
            if (300 <= response.status_code < 400) or (
                response.headers.get("Location", "").startswith("https://")
            ):
                self.server_address = self.server_address.replace("http://", "https://")
                msg = (
                    "You're using HTTP server address while the server requires HTTPS. "
                    "Supervisely automatically changed the server address to HTTPS for you. "
                    f"Consider updating your server address to {self.server_address}"
                )
                self.logger.warn(msg)

            self._require_https_redirect_check = False

    @classmethod
    def from_credentials(
        cls,
        server_address: str,
        login: str,
        password: str,
        override: bool = False,
        env_file: str = SUPERVISELY_ENV_FILE,
        check_instance_version: Union[bool, str] = False,
    ) -> Api:
        """
        Create Api object using credentials and optionally save them to ".env" file with overriding environment variables.
        If ".env" file already exists, backup will be created automatically.
        All backups will be stored in the same directory with postfix "_YYYYMMDDHHMMSS". You can have not more than 5 last backups.
        This method can be used also to update ".env" file.

        :param server_address: Supervisely server url.
        :type server_address: str
        :param login: User login.
        :type login: str
        :param password: User password.
        :type password: str
        :param override: If False, return Api object. If True, additionally create ".env" file or overwrite existing (backup file will be created automatically), and override environment variables.
        :type override: bool, optional
        :param env_file: Path to your .env file.
        :type env_file: str, optional
        :param check_instance_version: Check if the given version is lower or equal to the current
            version of the Supervisely instance.
        :type check_instance_version: bool or str, optional
        :return: Api object

        :Usage example:

             .. code-block:: python

                import supervisely as sly

                server_address = 'https://app.supervisely.com'
                login = 'user'
                password = 'pass'

                api = sly.Api.from_credentials(server_address, login, password)
        """

        session = UserSession(server_address).log_in(login, password)
        del password
        gc.collect()

        api = cls(
            session.server_address,
            session.api_token,
            ignore_task_id=True,
            check_instance_version=check_instance_version,
        )

        if override:
            if os.path.isfile(env_file):
                # create backup
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                backup_file = f"{env_file}_{timestamp}"
                shutil.copy2(env_file, backup_file)
                if api.token != get_key(env_file, API_TOKEN):
                    # create new file
                    os.remove(env_file)
                    Path(env_file).touch()
                # remove old backups
                all_backups = sorted(glob.glob(f"{env_file}_" + "[0-9]" * 14))
                while len(all_backups) > 5:
                    os.remove(all_backups.pop(0))
            set_key(env_file, SERVER_ADDRESS, session.server_address)
            set_key(env_file, API_TOKEN, session.api_token)
            if session.team_id:
                set_key(env_file, "INIT_GROUP_ID", f"{session.team_id}")
            if session.workspace_id:
                set_key(env_file, "INIT_WORKSPACE_ID", f"{session.workspace_id}")
            load_dotenv(env_file, override=override)
        return api

    @property
    def api_server_address(self) -> str:
        """
        Get API server address.

        :return: API server address.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api(server_address='https://app.supervisely.com', token='4r47N...xaTatb')
            print(api.api_server_address)
            # Output:
            # 'https://app.supervisely.com/public/api'
        """

        if self._api_server_address is not None:
            return self._api_server_address

        return f"{self.server_address}/public/api"
