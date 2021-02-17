# coding: utf-8

import os
import time
from collections import defaultdict, OrderedDict
import json

from supervisely_lib.api.module_api import ApiField, ModuleApiBase, ModuleWithStatus, WaitingTimeExceeded
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from supervisely_lib.io.fs import get_file_name, ensure_base_path, get_file_hash
from supervisely_lib.collection.str_enum import StrEnum
from supervisely_lib._utils import batched


class TaskApi(ModuleApiBase, ModuleWithStatus):
    class RestartPolicy(StrEnum):
        NEVER = 'never'
        ON_ERROR = 'on_error'

    class PluginTaskType(StrEnum):
        TRAIN = 'train'
        INFERENCE = 'inference'
        INFERENCE_RPC = 'inference_rpc'
        SMART_TOOL = 'smarttool'
        CUSTOM = 'custom'

    class Status(StrEnum):
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
        '''
        :param workspace_id: int
        :param filters: list
        :return: list of all tasks in given workspace
        '''
        return self.get_list_all_pages('tasks.list',
                                       {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        '''
        :param id: int
        :return: tast metadata by numeric id
        '''
        return self._get_info_by_id(id, 'tasks.info')

    def get_status(self, task_id):
        '''
        :param task_id: int
        :return: Status class object (status of task with given id)
        '''
        status_str = self.get_info_by_id(task_id)[ApiField.STATUS]  # @TODO: convert json to tuple
        return self.Status(status_str)

    def raise_for_status(self, status):
        '''
        Raise error if status is ERROR
        :param status: Status class object
        '''
        if status is self.Status.ERROR:
            raise RuntimeError('Task status: ERROR')

    def wait(self, id, target_status, wait_attempts=None, wait_attempt_timeout_sec=None):
        '''
        Awaiting achievement by given task of a given status
        :param id: int
        :param target_status: Status class object (status of task we expect to destinate)
        :param wait_attempts: int
        :param wait_attempt_timeout_sec: int (raise error if waiting time exceeded)
        :return: bool
        '''
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

    def deploy_model_async(self, agent_id, model_id):
        task_ids = self._api.model.get_deploy_tasks(model_id)
        if len(task_ids) == 0:
            task_id = self._deploy_model(agent_id, model_id)
        else:
            task_id = task_ids[0]
        return task_id

    def stop(self, id):
        response = self._api.post('tasks.stop', {ApiField.ID: id})
        return self.Status(response.json()[ApiField.STATUS])

    def get_import_files_list(self, id):
        response = self._api.post('tasks.import.files_list', {ApiField.ID: id})
        return response.json() if (response is not None) else None

    def download_import_file(self, id, file_path, save_path):
        response = self._api.post('tasks.import.download_file', {ApiField.ID: id, ApiField.FILENAME: file_path}, stream=True)

        ensure_base_path(save_path)
        with open(save_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def create_task_detached(self, workspace_id, task_type: str=None):
        response = self._api.post('tasks.run.python', {ApiField.WORKSPACE_ID: workspace_id,
                                                       ApiField.SCRIPT: "xxx",
                                                       ApiField.ADVANCED: {ApiField.IGNORE_AGENT: True}})
        return response.json()[ApiField.TASK_ID]

    def submit_logs(self, logs):
        response = self._api.post('tasks.logs.add', {ApiField.LOGS: logs})
        #return response.json()[ApiField.TASK_ID]

    def upload_files(self, task_id, abs_paths, names, progress_cb=None):
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
                resp = self._api.post('tasks.files.bulk.add-by-hash', {ApiField.TASK_ID: task_id, ApiField.FILES: batch})
        if progress_cb is not None:
            progress_cb(len(remote_hashes))

        for batch in batched(list(zip(abs_paths, names, hashes))):
            content_dict = OrderedDict()
            for idx, item in enumerate(batch):
                path, name, hash = item
                if hash in remote_hashes:
                    continue
                content_dict["{}".format(idx)] = json.dumps({"fullpath": name, "hash": hash})
                content_dict["{}-file".format(idx)] = (name, open(path, 'rb'), '')

            if len(content_dict) > 0:
                encoder = MultipartEncoder(fields=content_dict)
                resp = self._api.post('tasks.files.bulk.upload', encoder)
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

    def set_fields(self, task_id, fields):
        for idx, obj in enumerate(fields):
            for key in [ApiField.FIELD, ApiField.PAYLOAD]:
                if key not in obj:
                    raise KeyError("Object #{} does not have field {!r}".format(idx, key))
        data = {
            ApiField.TASK_ID: task_id,
            ApiField.FIELDS: fields
        }
        resp = self._api.post('tasks.data.set', data)
        return resp.json()

    def set_field(self, task_id, field, payload, append=False, recursive=False):
        fields = [
            {
                ApiField.FIELD: field,
                ApiField.PAYLOAD: payload,
                ApiField.APPEND: append,
                ApiField.RECURSIVE: recursive,
            }
        ]
        return self.set_fields(task_id, fields)

    def get_fields(self, task_id, fields: list) -> dict:
        data = {
            ApiField.TASK_ID: task_id,
            ApiField.FIELDS: fields
        }
        resp = self._api.post('tasks.data.get', data)
        return resp.json()["result"]

    def get_field(self, task_id, field):
        result = self.get_fields(task_id, [field])
        return result[field]

    def _validate_checkpoints_support(self, task_id):
        info = self.get_info_by_id(task_id)
        if info["type"] != str(TaskApi.PluginTaskType.TRAIN):
            raise RuntimeError("Task (id={!r}) has type {!r}. "
                               "Checkpoints are available only for tasks of type {!r}".format())

    def list_checkpoints(self, task_id):
        self._validate_checkpoints_support(task_id)
        resp = self._api.post('tasks.checkpoints.list', {ApiField.ID: task_id})
        return resp.json()

    def delete_unused_checkpoints(self, task_id):
        self._validate_checkpoints_support(task_id)
        resp = self._api.post("tasks.checkpoints.clear", {ApiField.ID: task_id})
        return resp.json()

    def _set_output(self):
        pass

    def set_output_project(self, task_id, project_id, project_name=None):
        if project_name is None:
            project = self._api.project.get_info_by_id(project_id)
            project_name = project.name

        output = {
            ApiField.PROJECT: {
                ApiField.ID: project_id,
                ApiField.TITLE: project_name
            }
        }
        resp = self._api.post("tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output})
        return resp.json()

    def set_output_report(self, task_id, file_id, file_name):
        return self._set_custom_output(task_id, file_id, file_name, description="Report", icon="zmdi zmdi-receipt")

    def _set_custom_output(self, task_id, file_id, file_name, file_url=None, description="File",
                           icon="zmdi zmdi-file-text", color="#33c94c", background_color="#d9f7e4",
                           download=False):
        if file_url is None:
            file_url = self._api.file.get_url(file_id)

        output = {
            ApiField.GENERAL: {
                "icon": {
                    "className": icon,
                    "color": color,
                    "backgroundColor": background_color
                },
                "title": file_name,
                "titleUrl": file_url,
                "download": download,
                "description": description
            }
        }
        resp = self._api.post("tasks.output.set", {ApiField.TASK_ID: task_id, ApiField.OUTPUT: output})
        return resp.json()

    def set_output_archive(self, task_id, file_id, file_name, file_url=None):
        if file_url is None:
            file_url = self._api.file.get_info_by_id(file_id).full_storage_url
        return self._set_custom_output(task_id, file_id, file_name,
                                       file_url=file_url,
                                       description="Download archive", icon="zmdi zmdi-archive",
                                       download=True)

    def send_request(self, task_id, method, data, context={}, skip_response=False):
        if type(data) is not dict:
            raise TypeError("data argument has to be a dict")
        resp = self._api.post("tasks.request.direct", {ApiField.TASK_ID: task_id,
                                                       ApiField.COMMAND: method,
                                                       ApiField.CONTEXT: context,
                                                       ApiField.STATE: data,
                                                       'skipResponse': skip_response})
        return resp.json()

    def set_output_directory(self, task_id, file_id, directory_path):
        return self._set_custom_output(task_id, file_id, directory_path, description="Directory", icon="zmdi zmdi-folder")