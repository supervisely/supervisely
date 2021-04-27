# coding: utf-8
from copy import deepcopy
import os
import os.path as osp
import threading
import time
import multiprocessing
import concurrent.futures
import sys
import json

import supervisely_lib as sly
from supervisely_lib.io.json import dump_json_file

from worker.data_manager import DataManager
from worker.logs_to_rpc import add_task_handler
from worker.agent_utils import LogQueue
from worker import constants


class StopTaskException(Exception):
    pass


exit_codes = {sly.EventType.TASK_FINISHED: 0,
              sly.EventType.TASK_CRASHED: 1,
              sly.EventType.TASK_STOPPED: 2}


# common process, where logs are streamed and saved; the task may be stopped
class TaskLogged(multiprocessing.Process):
    def __init__(self, task_info):
        super().__init__()
        self.daemon = True
        self.info = deepcopy(task_info)
        # Move API key out of the task info message so that it does not get into
        # logs.
        self._user_api_key = self.info.pop('user_api_key', None)

        self.init_task_dir()
        self.dir_logs = os.path.join(self.dir_task, 'logs')
        sly.fs.mkdir(self.dir_task)
        sly.fs.mkdir(self.dir_logs)

        self.logger = None
        self.log_queue = None
        self.executor_log = None
        self.future_log = None

        self._stop_log_event = None
        self._stop_event = multiprocessing.Event()

        # pre-init for static analysis
        self.api = None
        self.data_mgr = None
        self.public_api = None
        self.public_api_context = None

    def init_task_dir(self):
        self.dir_task = osp.join(constants.AGENT_TASKS_DIR(), str(self.info['task_id']))
        self.dir_task_host = osp.join(constants.AGENT_TASKS_DIR_HOST(), str(self.info['task_id']))

    def init_logger(self, loglevel=None):
        self.logger = sly.get_task_logger(self.info['task_id'], loglevel=loglevel)
        sly.change_formatters_default_values(self.logger, 'service_type', sly.ServiceType.AGENT)
        sly.change_formatters_default_values(self.logger, 'event_type', sly.EventType.LOGJ)

        self.log_queue = LogQueue()
        add_task_handler(self.logger, self.log_queue)
        sly.add_default_logging_into_file(self.logger, self.dir_logs)
        self.executor_log = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def init_api(self):
        self.api = sly.AgentAPI(constants.TOKEN(), constants.SERVER_ADDRESS(), self.logger, constants.TIMEOUT_CONFIG_PATH())

        if self._user_api_key is not None:
            self.public_api = sly.Api(constants.SERVER_ADDRESS(), self._user_api_key, external_logger=self.logger,
                                      retry_count=constants.PUBLIC_API_RETRY_LIMIT())
            task_id = self.info['task_id']
            self.public_api.add_additional_field('taskId', task_id)
            self.public_api.add_header('x-task-id', str(task_id))
            self.public_api_context = self.public_api.task.get_context(task_id)
        # else -> TelemetryReporter

    def init_additional(self):
        self.data_mgr = DataManager(self.logger, self.api, self.public_api, self.public_api_context)

    def submit_log(self):
        break_flag = False
        while True:
            log_lines = self.log_queue.get_log_batch_nowait()
            if len(log_lines) > 0:
                self.api.simple_request('Log', sly.api_proto.Empty, sly.api_proto.LogLines(data=log_lines))
                break_flag = False
            else:
                if break_flag:
                    return True
                if self._stop_log_event.isSet():
                    break_flag = True  # exit after next loop without data
                time.sleep(0.3)

    def end_log_stop(self):
        self.logger.info('TASK_END', extra={'event_type': sly.EventType.TASK_STOPPED, 'stopped': 'by_user'})
        return sly.EventType.TASK_STOPPED

    def end_log_crash(self, e):
        self.logger.critical('TASK_END', exc_info=True, extra={'event_type': sly.EventType.TASK_CRASHED, 'exc_str': str(e)})
        return sly.EventType.TASK_CRASHED

    def end_log_finish(self):
        self.logger.info('TASK_END', extra={'event_type': sly.EventType.TASK_FINISHED})
        return sly.EventType.TASK_FINISHED

    # in new process
    def run(self):
        try:
            self._stop_log_event = threading.Event()
            self.init_logger()
            self.init_api()
            self.future_log = self.executor_log.submit(self.submit_log)  # run log submitting
        except Exception as e:
            # unable to do something another if crashed
            print(e)
            dump_json_file(str(e), os.path.join(constants.AGENT_ROOT_DIR(), 'logger_fail.json'))
            os._exit(1)  # ok, documented

        try:
            self.report_start()
            self.init_additional()
            self.run_and_wait(self.task_main_func)
        except StopTaskException:
            exit_status = self.end_log_stop()
        except Exception as e:
            exit_status = self.end_log_crash(e)
        else:
            exit_status = self.end_log_finish()

        self.logger.info("WAIT_FOR_TASK_LOG")
        self.stop_log_thread()
        self.close_log_handlers()

        sys.exit(exit_codes[exit_status])

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

    def close_log_handlers(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
