# coding: utf-8

import supervisely_lib as sly

from supervisely_lib._utils import _remove_sensitive_information
from worker.task_logged import TaskLogged
import logging


# a task that should be shown as a 'task' in web
class TaskSly(TaskLogged):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_logger(self, loglevel=None):
        super().init_logger(loglevel=loglevel)
        sly.change_formatters_default_values(self.logger, 'service_type', sly.ServiceType.TASK)
        sly.change_formatters_default_values(self.logger, 'task_id', self.info['task_id'])
        self.logger.info(f"TASK LOG LEVEL: {logging.getLevelName(self.logger.level)} ({self.logger.level})")

    def init_api(self):
        super().init_api()
        self.api.add_to_metadata('x-task-id', str(self.info['task_id']))

    def report_start(self):
        self.logger.info('TASK_START', extra={'event_type': sly.EventType.TASK_STARTED})
        to_log = _remove_sensitive_information(self.info)
        self.logger.info('TASK_MSG', extra=to_log)

    def task_main_func(self):
        raise NotImplementedError()
