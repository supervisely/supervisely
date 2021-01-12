# coding: utf-8

import os
import requests
import json

from requests_toolbelt import MultipartEncoderMonitor, MultipartEncoder

import supervisely_lib.api.team_api as team_api
import supervisely_lib.api.workspace_api as workspace_api
import supervisely_lib.api.project_api as project_api
import supervisely_lib.api.neural_network_api as neural_network_api
import supervisely_lib.api.task_api as task_api
import supervisely_lib.api.dataset_api as dataset_api
import supervisely_lib.api.image_api as image_api
import supervisely_lib.api.annotation_api as annotation_api
import supervisely_lib.api.plugin_api as plugin_api
import supervisely_lib.api.agent_api as agent_api
import supervisely_lib.api.role_api as role_api
import supervisely_lib.api.user_api as user_api
import supervisely_lib.api.labeling_job_api as labeling_job_api
import supervisely_lib.api.video.video_api as video_api
import supervisely_lib.api.pointcloud.pointcloud_api as pointcloud_api
import supervisely_lib.api.object_class_api as object_class_api
import supervisely_lib.api.report_api as report_api
import supervisely_lib.api.app_api as app_api
import supervisely_lib.api.file_api as file_api
import supervisely_lib.api.image_annotation_tool_api as image_annotation_tool_api
import supervisely_lib.api.advanced_api as advanced_api
import supervisely_lib.api.import_storage_api as import_stoarge_api

from supervisely_lib.sly_logger import logger


from supervisely_lib.io.network_exceptions import process_requests_exception, process_unhandled_request

SUPERVISELY_TASK_ID = 'SUPERVISELY_TASK_ID'
SUPERVISELY_PUBLIC_API_RETRIES = 'SUPERVISELY_PUBLIC_API_RETRIES'
SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC = 'SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC'
SERVER_ADDRESS = 'SERVER_ADDRESS'
API_TOKEN = 'API_TOKEN'


class Api:
    def __init__(self, server_address, token, retry_count=None, retry_sleep_sec=None, external_logger=None, ignore_task_id=False):
        '''
        :param server_address: str (example: http://192.168.1.69:5555)
        :param token: str
        :param retry_count: int
        :param retry_sleep_sec: int
        :param external_logger: logger class object
        '''
        if token is None:
            raise ValueError("Token is None")
        self.server_address = server_address.strip('/')
        if ('http://' not in self.server_address) and ('https://' not in self.server_address):
            self.server_address = 'http://' + self.server_address

        if retry_count is None:
            retry_count = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRIES, '20'))
        if retry_sleep_sec is None:
            retry_sleep_sec = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC, '1'))

        if len(token) != 128:
            raise ValueError("Invalid token {!r}: length != 128".format(token))

        self.headers = {'x-api-key': token}
        self.task_id = os.getenv(SUPERVISELY_TASK_ID)
        if self.task_id is not None and ignore_task_id is False:
            self.headers['x-task-id'] = self.task_id
        self.context = {}
        self.additional_fields = {}

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
        self.app = app_api.AppApi(self)
        self.file = file_api.FileApi(self)
        self.img_ann_tool = image_annotation_tool_api.ImageAnnotationToolApi(self)
        self.advanced = advanced_api.AdvancedApi(self)
        self.import_storage = import_stoarge_api.ImportStorageApi(self)

        self.retry_count = retry_count
        self.retry_sleep_sec = retry_sleep_sec

        self.logger = external_logger or logger

    @classmethod
    def from_env(cls, retry_count=5, ignore_task_id=False):
        '''
        :return: Api class object with server adress and token obtained from environment variables
        '''
        return cls(os.environ[SERVER_ADDRESS], os.environ[API_TOKEN], retry_count=retry_count, ignore_task_id=ignore_task_id)

    def add_header(self, key, value):
        '''
        Add given key and value to headers dictionary. Raise error if key is already set.
        :param key: str
        :param value: str
        '''
        if key in self.headers:
            raise RuntimeError(f'Header {key!r} is already set for the API object. '
                               f'Current value: {self.headers[key]!r}. Tried to set value: {value!r}')
        self.headers[key] = value

    def add_additional_field(self, key, value):
        '''
        Add given key and value to additional_fields dictionary.
        :param key: str
        :param value: str
        '''
        self.additional_fields[key] = value

    def post(self, method, data, retries=None, stream=False):
        '''
        Performs POST request to server with given parameters. Raise error if can not connect to server.
        :param method: str
        :param data: dict
        :param retries: int (number of attempts to access the server)
        :param stream: bool
        :return: Request class object
        '''
        if retries is None:
            retries = self.retry_count

        for retry_idx in range(retries):
            url = self.server_address + '/public/api/v3/' + method
            response = None
            try:
                if type(data) is bytes:
                    response = requests.post(url, data=data, headers=self.headers, stream=stream)
                elif type(data) is MultipartEncoderMonitor or type(data) is MultipartEncoder:
                    response = requests.post(url, data=data,
                                             headers={**self.headers, 'Content-Type': data.content_type}, stream=stream)
                else:
                    json_body = data
                    if type(data) is dict:
                        json_body = {**data, **self.additional_fields}
                    response = requests.post(url, json=json_body, headers=self.headers, stream=stream)

                if response.status_code != requests.codes.ok:
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(self.logger, exc, method, url,
                                           verbose=True, swallow_exc=True, sleep_sec=self.retry_sleep_sec,
                                           response=response,
                                           retry_info={"retry_idx": retry_idx + 1,
                                                       "retry_limit": retries})
            except Exception as exc:
                process_unhandled_request(self.logger, exc)
        raise requests.exceptions.RetryError("Retry limit exceeded ({!r})".format(url))

    def get(self, method, params, retries=None, stream=False, use_public_api=True):
        '''
        Performs GET request to server with given parameters. Raise error if can not connect to server.
        :param method: str
        :param params: dict
        :param retries: int (number of attempts to access the server)
        :param stream: bool
        :param use_public_api: bool
        :return: Request class object
        '''
        if retries is None:
            retries = self.retry_count

        for retry_idx in range(retries):
            url = self.server_address + '/public/api/v3/' + method
            if use_public_api is False:
                url = os.path.join(self.server_address, method)
            response = None
            try:
                json_body = params
                if type(params) is dict:
                    json_body = {**params, **self.additional_fields}
                response = requests.get(url, params=json_body, headers=self.headers, stream=stream)

                if response.status_code != requests.codes.ok:
                    Api._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                process_requests_exception(self.logger, exc, method, url,
                                           verbose=True, swallow_exc=True, sleep_sec=self.retry_sleep_sec,
                                           response=response,
                                           retry_info={"retry_idx": retry_idx + 2,
                                                       "retry_limit": retries})
            except Exception as exc:
                process_unhandled_request(self.logger, exc)

    @staticmethod
    def _raise_for_status(response):
        '''
        Raise error and show message with code of mistake if given response can not connect to server.
        :param response: Request class object
        '''
        """Raises stored :class:`HTTPError`, if one occurred."""
        http_error_msg = ''
        if isinstance(response.reason, bytes):
            try:
                reason = response.reason.decode('utf-8')
            except UnicodeDecodeError:
                reason = response.reason.decode('iso-8859-1')
        else:
            reason = response.reason

        if 400 <= response.status_code < 500:
            http_error_msg = u'%s Client Error: %s for url: %s (%s)' % (response.status_code, reason, response.url, response.content.decode('utf-8'))

        elif 500 <= response.status_code < 600:
            http_error_msg = u'%s Server Error: %s for url: %s (%s)' % (response.status_code, reason, response.url, response.content.decode('utf-8'))

        if http_error_msg:
            raise requests.exceptions.HTTPError(http_error_msg, response=response)

    @staticmethod
    def parse_error(response, default_error="Error", default_message="please, contact administrator"):
        '''

        :param response: Request class object
        :param default_error: str
        :param default_message: str
        :return: number of error and message about curren connection mistake
        '''
        ERROR_FIELD = "error"
        MESSAGE_FIELD = "message"
        DETAILS_FIELD = "details"

        try:
            data_str = response.content.decode('utf-8')
            data = json.loads(data_str)
            error = data.get(ERROR_FIELD, default_error)
            details = data.get(DETAILS_FIELD, {})
            if type(details) is dict:
                message = details.get(MESSAGE_FIELD, default_message)
            else:
                message = details[0].get(MESSAGE_FIELD, default_message)

            return error, message
        except Exception as e:
            return "", ""
