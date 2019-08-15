# coding: utf-8

from enum import Enum
import os
import time

from supervisely_lib.api.module_api import ApiField, ModuleApiBase, ModuleWithStatus, WaitingTimeExceeded
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from supervisely_lib.io.fs import get_file_name


class TaskApi(ModuleApiBase, ModuleWithStatus):
    class RestartPolicy(Enum):
        NEVER = 'never'
        ON_ERROR = 'on_error'

    class PluginTaskType(Enum):
        TRAIN = 'train'
        INFERENCE = 'inference'
        INFERENCE_RPC = 'inference_rpc'
        SMART_TOOL = 'smarttool'
        CUSTOM = 'custom'

    class Status(Enum):
        QUEUED = 'queued'
        CONSUMED = 'consumed'
        STARTED = 'started'
        DEPLOYED = 'deployed'
        ERROR = 'error'
        FINISHED = 'finished'
        TERMINATING = 'terminating'
        STOPPED = 'stopped'

    def __init__(self, api):
        ModuleApiBase.__init__(self, api)
        ModuleWithStatus.__init__(self)

    def get_list(self, workspace_id, filters=None):
        return self.get_list_all_pages('tasks.list',
                                       {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'tasks.info')

    def get_status(self, task_id):
        status_str = self.get_info_by_id(task_id)[ApiField.STATUS]  # @TODO: convert json to tuple
        return self.Status(status_str)

    def raise_for_status(self, status):
        if status is self.Status.ERROR:
            raise RuntimeError('Task status: ERROR')

    def wait(self, id, target_status, wait_attempts=None, wait_attempt_timeout_sec=None):
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status is target_status:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded('Waiting time exceeded')

    def upload_dtl_archive(self, task_id, archive_path, progress_cb=None):
        encoder = MultipartEncoder({'id': str(task_id).encode('utf-8'),
                                    'name': get_file_name(archive_path),
                                    'archive': (
                                    os.path.basename(archive_path), open(archive_path, 'rb'), 'application/x-tar')})

        def callback(monitor_instance):
            read_mb = monitor_instance.bytes_read / 1024.0 / 1024.0
            if progress_cb is not None:
                progress_cb(read_mb)

        monitor = MultipartEncoderMonitor(encoder, callback)
        self._api.post('tasks.upload.dtl_archive', monitor)

    def _deploy_model(self, agent_id, model_id, plugin_id=None, version=None, restart_policy=RestartPolicy.NEVER,
                      settings=None):
        response = self._api.post('tasks.run.deploy', {ApiField.AGENT_ID: agent_id,
                                                       ApiField.MODEL_ID: model_id,
                                                       ApiField.RESTART_POLICY: restart_policy.value,
                                                       ApiField.SETTINGS: settings or {'gpu_device': 0},
                                                       ApiField.PLUGIN_ID: plugin_id,
                                                       ApiField.VERSION: version})
        return response.json()[ApiField.TASK_ID]

    def get_context(self, id):
        response = self._api.post('GetTaskContext', {ApiField.ID: id})
        return response.json()

    def _convert_json_info(self, info: dict):
        return info

    def run_dtl(self, workspace_id, dtl_graph, agent_id=None):
        response = self._api.post('tasks.run.dtl', {ApiField.WORKSPACE_ID: workspace_id,
                                                    ApiField.CONFIG: dtl_graph,
                                                    'advanced': {ApiField.AGENT_ID: agent_id}})
        return response.json()[ApiField.TASK_ID]

    def _run_plugin_task(self, task_type, agent_id, plugin_id, version, config, input_projects, input_models,
                         result_name):
        response = self._api.post('tasks.run.plugin', {'taskType': task_type,
                                                       ApiField.AGENT_ID: agent_id,
                                                       ApiField.PLUGIN_ID: plugin_id,
                                                       ApiField.VERSION: version,
                                                       ApiField.CONFIG: config,
                                                       'projects': input_projects,
                                                       'models': input_models,
                                                       ApiField.NAME: result_name})
        return response.json()[ApiField.TASK_ID]

    def run_train(self, agent_id, input_project_id, input_model_id, result_nn_name, train_config=None):
        model_info = self._api.model.get_info_by_id(input_model_id)
        return self._run_plugin_task(task_type=TaskApi.PluginTaskType.TRAIN.value,
                                     agent_id=agent_id,
                                     plugin_id=model_info.plugin_id,
                                     version=None,
                                     input_projects=[input_project_id],
                                     input_models=[input_model_id],
                                     result_name=result_nn_name,
                                     config={} if train_config is None else train_config)

    def run_inference(self, agent_id, input_project_id, input_model_id, result_project_name, inference_config=None):
        model_info = self._api.model.get_info_by_id(input_model_id)
        return self._run_plugin_task(task_type=TaskApi.PluginTaskType.INFERENCE.value,
                                     agent_id=agent_id,
                                     plugin_id=model_info.plugin_id,
                                     version=None,
                                     input_projects=[input_project_id],
                                     input_models=[input_model_id],
                                     result_name=result_project_name,
                                     config={} if inference_config is None else inference_config)

    def get_training_metrics(self, task_id):
        response = self._get_response_by_id(id=task_id, method='tasks.train-metrics', id_field=ApiField.TASK_ID)
        return response.json() if (response is not None) else None

    def deploy_model(self, agent_id, model_id):
        task_ids = self._api.model.get_deploy_tasks(model_id)
        if len(task_ids) == 0:
            task_id = self._deploy_model(agent_id, model_id)
        else:
            task_id = task_ids[0]
        self.wait(task_id, self.Status.DEPLOYED)
        return task_id

    def stop(self, id):
        response = self._api.post('tasks.stop', {ApiField.ID: id})
        return self.Status(response.json()[ApiField.STATUS])