# coding: utf-8

from supervisely_lib import logger

from .task_dockerized import TaskDockerized
from .task_dtl import TaskDTL
from .task_import import TaskImport
from .task_upload_nn import TaskUploadNN
from .task_train import TaskTrain
from .task_inference import TaskInference
from .task_clean_node import TaskCleanNode
from .task_inference_rpc import TaskInferenceRPC
from .task_upload_images import TaskUploadImages
from .task_import_local import TaskImportLocal


_task_class_mapping = {
    'export':        TaskDTL,
    'import':        TaskImport,
    'upload_model':  TaskUploadNN,
    'train':         TaskTrain,
    'inference':     TaskInference,
    'cleanup':       TaskCleanNode,
    'smarttool':     TaskInferenceRPC,  # for compatibility
    'infer_rpc':     TaskInferenceRPC,
    'upload_images': TaskUploadImages,
    'import_agent':  TaskImportLocal,
}


def create_task(task_msg, docker_api):
    task_cls = _task_class_mapping.get(task_msg['task_type'], None)
    if task_cls is None:
        logger.critical('unknown task type', extra={'task_msg': task_msg})
        raise RuntimeError('unknown task type')

    task_obj = task_cls(task_msg)
    if issubclass(task_cls, TaskDockerized):
        task_obj.docker_api = docker_api
    return task_obj
