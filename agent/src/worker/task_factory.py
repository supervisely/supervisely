# coding: utf-8

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
    'update_agent':         TaskUpdate
}


def create_task(task_msg, docker_api):
    task_cls = _task_class_mapping.get(task_msg['task_type'], None)
    if task_cls is None:
        sly.logger.critical('unknown task type', extra={'task_msg': task_msg})
        raise RuntimeError('unknown task type')
    task_obj = task_cls(task_msg)
    if issubclass(task_cls, TaskDockerized) or (task_msg['task_type'] == 'update_agent'):
        task_obj.docker_api = docker_api
    return task_obj
