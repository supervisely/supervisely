# coding: utf-8

import logging

from supervisely_lib.sly_logger import add_logger_handler


class TaskHandler(logging.Handler):
    def __init__(self, task_log_queue):
        super(TaskHandler, self).__init__()
        self.task_log_queue = task_log_queue

    def emit(self, record):
        log_entry = self.format(record)
        self.task_log_queue.put_nowait(log_entry)


def add_task_handler(the_logger, task_log_queue):
    task_handler = TaskHandler(task_log_queue)
    add_logger_handler(the_logger, task_handler)
