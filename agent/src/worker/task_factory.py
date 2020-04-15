# coding: utf-8

import base64
import json
from docker.errors import DockerException, ImageNotFound

import supervisely_lib as sly

from worker.task_dockerized import TaskDockerized
from worker.task_dtl import TaskDTL
from worker.task_import import TaskImport
from worker.task_upload_nn import TaskUploadNN
from worker.task_train import TaskTrain
from worker.task_inference import TaskInference
from worker.task_clean_node import TaskCleanNode
from worker.task_inference_rpc import TaskInferenceRPC
from worker.task_upload_images import TaskUploadImages
from worker.task_import_local import TaskImportLocal
from worker.task_custom import TaskCustom
from worker.task_update import TaskUpdate
from worker.task_python import TaskPython
from worker.task_plugin import TaskPlugin
from worker.task_plugin_import_local import TaskPluginImportLocal


_task_class_mapping = {
    'export':               TaskDTL,
    'import':               TaskImport,
    'upload_model':         TaskUploadNN,
    'train':                TaskTrain,
    'inference':            TaskInference,
    'cleanup':              TaskCleanNode,
    'smarttool':            TaskInferenceRPC,  # for compatibility
    'infer_rpc':            TaskInferenceRPC,
    'upload_images':        TaskUploadImages,
    'import_agent':         TaskImportLocal,
    'custom': 				TaskCustom,
    'update_agent':         TaskUpdate,
    'python':               TaskPython,
    'general_plugin':       TaskPlugin,
    'general_plugin_import_agent': TaskPluginImportLocal
}


def create_task(task_msg, docker_api):
    task_type = get_run_mode(docker_api, task_msg)
    task_cls = _task_class_mapping.get(task_type, None)
    if task_cls is None:
        sly.logger.critical('unknown task type', extra={'task_msg': task_msg})
        raise RuntimeError('unknown task type')
    task_obj = task_cls(task_msg)
    if issubclass(task_cls, TaskDockerized) or (task_msg['task_type'] == 'update_agent'):
        task_obj.docker_api = docker_api
    return task_obj


def get_run_mode(docker_api, task_msg):
    if "docker_image" not in task_msg:
        return task_msg['task_type']

    try:
        image_info = docker_api.images.get(task_msg["docker_image"])
    except ImageNotFound:
        image_info = {}
    except DockerException:
        image_info = docker_api.images.pull(task_msg["docker_image"])

    try:
        plugin_info = json.loads(base64.b64decode(image_info.labels["INFO"]).decode("utf-8"))
    except Exception as e:
        plugin_info = {}

    result = plugin_info.get("run_mode", task_msg['task_type'])

    if result == 'general_plugin' and task_msg['task_type'] == "import_agent":
        return 'general_plugin_import_agent'

    return result
