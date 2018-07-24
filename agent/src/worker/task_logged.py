# coding: utf-8

import os
import os.path as osp
import traceback
import threading
import time
import multiprocessing
import concurrent.futures

import supervisely_lib as sly
from supervisely_lib import ServiceType, EventType, get_task_logger
from supervisely_lib.worker_api import AgentAPI
import supervisely_lib.worker_proto as api_proto

from .data_manager import DataManager
from .logs_to_rpc import add_task_handler
from .agent_utils import LogQueue
from . import constants


class StopTaskException(Exception):
    pass


# common process, where logs are streamed and saved; the task may be stopped
class TaskLogged(multiprocessing.Process):
    def __init__(self, task_info):
        super().__init__()
        self.daemon = True
        self.info = task_info

        self.dir_task = osp.join(constants.AGENT_TASKS_DIR, str(self.info['task_id']))
        self.dir_logs = osp.join(self.dir_task, 'logs')
        sly.mkdir(self.dir_task)
        sly.mkdir(self.dir_logs)
        self.dir_task_host = osp.join(constants.AGENT_TASKS_DIR_HOST, str(self.info['task_id']))

        self._stop_log_event = threading.Event()
        self._stop_event = multiprocessing.Event()

        # pre-init for static analysis
        self.logger = None
        self.log_queue = None
        self.executor_log = None
        self.future_log = None

        self.api = None
        self.data_mgr = None

    def init_logger(self):
        self.logger = get_task_logger(self.info['task_id'])
        sly.change_formatters_default_values(self.logger, 'service_type', ServiceType.AGENT)
        sly.change_formatters_default_values(self.logger, 'event_type', EventType.LOGJ)

        self.log_queue = LogQueue()
        add_task_handler(self.logger, self.log_queue)
        sly.add_default_logging_into_file(self.logger, self.dir_logs)

        self.executor_log = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def init_api(self):
        self.api = AgentAPI(constants.TOKEN, constants.SERVER_ADDRESS, self.logger)

    def init_additional(self):
        self.data_mgr = DataManager(self.logger, self.api)

    def submit_log(self):
        break_flag = False
        while True:
            log_lines = self.log_queue.get_log_batch_nowait()
            if len(log_lines) > 0:
                self.api.simple_request('Log', api_proto.Empty, api_proto.LogLines(data=log_lines))
                break_flag = False
            else:
                if break_flag:
                    return True
                if self._stop_log_event.isSet():
                    break_flag = True  # exit after next loop without data
                time.sleep(0.3)

    # in new process
    def run(self):
        try:
            self.init_logger()
            self.init_api()
            self.future_log = self.executor_log.submit(self.submit_log)  # run log submitting
        except Exception as e:
            # unable to do something another if crashed
            print(e)
            sly.json_dump(e, osp.join(constants.AGENT_ROOT_DIR, 'logger_fail.json'))
            os._exit(1)  # ok, documented

        log_extra = {'event_type': EventType.TASK_FINISHED}
        logger_fn = self.logger.info
        try:
            self.report_start()
            self.init_additional()
            self.run_and_wait(self.task_main_func)
        except StopTaskException:
            log_extra = {'event_type': EventType.TASK_STOPPED, 'stopped': 'by_user'}
        except Exception:
            log_extra = {'event_type': EventType.TASK_CRASHED, 'error': traceback.format_exc().split('\n')}
            logger_fn = self.logger.critical
        logger_fn('TASK_END',  extra=log_extra)

        self.logger.info("WAIT_FOR_TASK_LOG")
        self.stop_log_thread()

    def task_main_func(self):
        raise NotImplementedError()

    def run_and_wait(self, subtask_fn):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(subtask_fn)
        while future.running():
            time.sleep(0.5)
            if self._stop_event.is_set():
                executor.shutdown(wait=False)
                raise StopTaskException()

            if not self.future_log.running():
                raise RuntimeError("SUBMIT_LOGS_ERROR")

        executor.shutdown(wait=True)
        return future.result()

    def stop_log_thread(self):
        self._stop_log_event.set()
        self.executor_log.shutdown(wait=True)
        return self.future_log.result()  # crash if log thread crashed

    def join(self, timeout=None):
        self._stop_event.set()
        super().join(timeout)

    def report_start(self):
        pass

    def clean_task_dir(self):
        pass
