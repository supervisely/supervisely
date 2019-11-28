# coding: utf-8

import os
import requests

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
from supervisely_lib.sly_logger import logger


from supervisely_lib.io.network_exceptions import process_requests_exception, process_unhandled_request

SUPERVISELY_TASK_ID = 'SUPERVISELY_TASK_ID'
SUPERVISELY_PUBLIC_API_RETRIES = 'SUPERVISELY_PUBLIC_API_RETRIES'
SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC = 'SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC'


class Api:
    def __init__(self, server_address, token, retry_count=None, retry_sleep_sec=None, external_logger=None):
        if token is None:
            raise ValueError("Token is None")
        self.server_address = server_address.strip('/')
        if ('http://' not in self.server_address) and ('https://' not in self.server_address):
            self.server_address = 'http://' + self.server_address

        if retry_count is None:
            retry_count = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRIES, '7000'))
        if retry_sleep_sec is None:
            retry_sleep_sec = int(os.getenv(SUPERVISELY_PUBLIC_API_RETRY_SLEEP_SEC, '1'))

        self.headers = {'x-api-key': token}
        task_id = os.getenv(SUPERVISELY_TASK_ID)
        if task_id is not None:
            self.headers['x-task-id'] = task_id
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

        self.retry_count = retry_count
        self.retry_sleep_sec = retry_sleep_sec

        self.logger = external_logger or logger

    def add_header(self, key, value):
        if key in self.headers:
            raise RuntimeError(f'Header {key!r} is already set for the API object. '
                               f'Current value: {self.headers[key]!r}. Tried to set value: {value!r}')
        self.headers[key] = value

    def add_additional_field(self, key, value):
        self.additional_fields[key] = value

    def post(self, method, data, retries=None, stream=False):
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
                                           retry_info={"retry_idx": retry_idx + 2,
                                                       "retry_limit": retries})
            except Exception as exc:
                process_unhandled_request(self.logger, exc)

    @staticmethod
    def _raise_for_status(response):
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
