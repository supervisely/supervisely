# coding: utf-8

import os

from ..utils import json_load
from ..project import ProjectMeta
from ..sly_logger import logger, EventType
from .task_paths import TaskPaths
from .train_checkpoints import TrainCheckpoints


class TaskHelperTrain:
    def __init__(self):
        self.paths = TaskPaths()
        self.task_settings = json_load(self.paths.settings_path)
        self.in_project_meta = ProjectMeta.from_dir(self.paths.project_dir)
        self.checkpoints_saver = TrainCheckpoints(self.paths.results_dir)

    def model_dir_is_empty(self):
        mdir = self.paths.model_dir
        res = not (os.path.isdir(mdir) and len(os.listdir(mdir)) > 0)
        return res


class TaskHelperInference:
    def __init__(self):
        self.paths = TaskPaths()
        self.task_settings = json_load(self.paths.settings_path)
        self.in_project_meta = ProjectMeta.from_dir(self.paths.project_dir)


def task_verification(verification_fn, *args, **kwargs):
    if os.getenv('VERIFICATION') is None:
        return
    logger.info('Verification started.')
    res = verification_fn(*args, **kwargs)
    logger.info('Verification finished.', extra={'output': res, 'event_type': EventType.TASK_VERIFIED})
    os._exit(0)
