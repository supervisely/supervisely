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


class Api:
    def __init__(self, server_address, token):
        if token is None:
            raise ValueError("Token is None")
        self.server_address = server_address
        if ('http://' not in self.server_address) and ('https://' not in self.server_address):
            self.server_address = os.path.join('http://', self.server_address)

        self.headers = {'x-api-key': token}
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

    def add_additional_field(self, key, value):
        self.additional_fields[key] = value

    def post(self, method, data):
        url = os.path.join(self.server_address, 'public/api/v3', method)

        if type(data) is bytes:
            response = requests.post(url, data=data, headers=self.headers)
        elif type(data) is MultipartEncoderMonitor or type(data) is MultipartEncoder:
            response = requests.post(url, data=data, headers={**self.headers, 'Content-Type': data.content_type})
        else:
            json_body = data
            if type(data) is dict:
                json_body = {**data, **self.additional_fields}
            response = requests.post(url, json=json_body, headers=self.headers)

        if response.status_code != requests.codes.ok:
            Api._raise_for_status(response)
        return response

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