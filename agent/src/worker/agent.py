# coding: utf-8

import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import docker
import supervisely_lib as sly
from supervisely_lib.sly_logger import EventType, ServiceType, get_task_logger
from supervisely_lib.worker_api import AgentAPI
import supervisely_lib.worker_proto as api_proto
from supervisely_lib.utils.docker_utils import remove_containers
from supervisely_lib import function_wrapper

from . import constants
from .task_factory import create_task
from .logs_to_rpc import add_task_handler
from .agent_utils import LogQueue
from .system_info import get_hw_info, get_self_docker_image_digest
from .image_streamer import ImageStreamer
from .telemetry_reporter import TelemetryReporter


class Agent:
    def __init__(self):
        self.logger = get_task_logger('agent')
        sly.change_formatters_default_values(self.logger, 'service_type', ServiceType.AGENT)
        sly.change_formatters_default_values(self.logger, 'event_type', EventType.LOGJ)
        self.log_queue = LogQueue()
        add_task_handler(self.logger, self.log_queue)
        sly.add_default_logging_into_file(self.logger, constants.AGENT_LOG_DIR)

        self.task_pool_lock = threading.Lock()
        self.task_pool = {}  # task_id -> task_manager (process_id)

        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.thread_list = []

        self.api = AgentAPI(constants.TOKEN, constants.SERVER_ADDRESS, self.logger)
        self.agent_connect_initially()

        sly.clean_dir(constants.AGENT_TMP_DIR)
        self._stop_missed_containers()

        self.docker_api = docker.from_env(version='auto')
        self._docker_login()

        self.logger.info('Agent is ready to get tasks.', extra={'event_type': EventType.AGENT_READY_FOR_TASKS})

    def agent_connect_initially(self):
        agent_info = {
            'hardware_info': get_hw_info(),
            'agent_image_digest': get_self_docker_image_digest()
        }

        self.logger.info('Will connect to server.')
        self.api.simple_request('AgentConnected', api_proto.ServerInfo,
                                api_proto.AgentInfo(info=sly.json_dumps(agent_info)))
        self.logger.info('Connected to server.')

    def get_new_task(self):
        for task in self.api.get_endless_stream('GetNewTask', api_proto.Task, api_proto.Empty()):
            task_msg = sly.json_loads(task.data)
            self.logger.debug('GET_NEW_TASK', extra={'task_msg': task_msg})
            self.start_task(task_msg)

    def get_stop_task(self):
        for task in self.api.get_endless_stream('GetStopTask', api_proto.Id, api_proto.Empty()):
            stop_task_id = task.id
            self.logger.debug('GET_STOP_TASK', extra={'task_id': stop_task_id})
            self.stop_task(stop_task_id)

    def stop_task(self, task_id):
        self.task_pool_lock.acquire()
        try:
            if task_id in self.task_pool:
                self.task_pool[task_id].join(timeout=20)
                self.task_pool[task_id].terminate()

                task_extra = {'task_id': task_id,
                              'exit_status': self.task_pool[task_id],
                              'exit_code': self.task_pool[task_id].exitcode}

                self.logger.info('REMOVE_TASK_TEMP_DATA IF NECESSARY', extra=task_extra)
                self.task_pool[task_id].clean_task_dir()

                self.logger.info('TASK_STOPPED', extra=task_extra)
                del self.task_pool[task_id]

            else:
                self.logger.warning('Task could not be stopped. Not found', extra={'task_id': task_id})

                self.logger.info('TASK_MISSED', extra={'service_type': ServiceType.TASK,
                                                       'event_type': EventType.TASK_STOPPED,
                                                       'task_id': task_id})

        finally:
            self.task_pool_lock.release()

    def start_task(self, task):
        self.task_pool_lock.acquire()
        try:
            if task['task_id'] in self.task_pool:
                self.logger.warning('TASK_ID_ALREADY_STARTED', extra={'task_id': task['task_id']})
            else:
                task_id = task['task_id']
                self.task_pool[task_id] = create_task(task, self.docker_api)
                self.task_pool[task_id].start()
        finally:
            self.task_pool_lock.release()

    def tasks_health_check(self):
        while True:
            time.sleep(3)
            self.task_pool_lock.acquire()
            try:
                all_tasks = list(self.task_pool.keys())
                for task_id in all_tasks:
                    val = self.task_pool[task_id]
                    if not val.is_alive():
                        self._forget_task(task_id)
            finally:
                self.task_pool_lock.release()

    # used only in healthcheck
    def _forget_task(self, task_id):
        task_extra = {'event_type': EventType.TASK_REMOVED,
                      'task_id': task_id,
                      'exit_status': self.task_pool[task_id],
                      'exit_code': self.task_pool[task_id].exitcode,
                      'service_type': ServiceType.TASK}

        self.logger.info('REMOVE_TASK_TEMP_DATA IF NECESSARY', extra=task_extra)
        self.task_pool[task_id].clean_task_dir()

        del self.task_pool[task_id]
        self.logger.info('TASK_REMOVED', extra=task_extra)

    def _stop_missed_containers(self):
        self.logger.info('Searching missed containers ...')
        label_filter = {'label': 'ecosystem_token={}'.format(constants.TASKS_DOCKER_LABEL)}
        stopped_list = remove_containers(label_filter=label_filter)
        if len(stopped_list) == 0:
            self.logger.info('There are no missed containers')

        for cont in stopped_list:
            self.logger.info('Container stopped', extra={'cont_id': cont.id, 'labels': cont.labels})
            self.logger.info('TASK_MISSED', extra={'service_type': ServiceType.TASK,
                                                   'event_type': EventType.TASK_CRASHED,
                                                   'task_id': int(cont.labels['task_id'])})

    def _docker_login(self):
        doc_logs = constants.DOCKER_LOGIN.split(',')
        doc_pasws = constants.DOCKER_PASSWORD.split(',')
        doc_regs = constants.DOCKER_REGISTRY.split(',')

        self.logger.info('Before Docker login.')
        for login, password, registry in zip(doc_logs, doc_pasws, doc_regs):
            doc_login = self.docker_api.login(username=login, password=password, registry=registry)
            self.logger.info('DOCKER_CLIENT_LOGIN_SUCCESS', extra={**doc_login, 'registry': registry})

    def submit_log(self):
        while True:
            log_lines = self.log_queue.get_log_batch_blocking()
            self.api.simple_request('Log', api_proto.Empty, api_proto.LogLines(data=log_lines))

    @classmethod
    def follow_daemon(cls, process_cls, name, sleep_sec=5):
        proc = process_cls()
        proc.start()
        while True:
            if not proc.is_alive():
                raise RuntimeError('{}_CRASHED'.format(name))
            time.sleep(sleep_sec)

    def inf_loop(self):
        self.thread_list.append(self.thread_pool.submit(function_wrapper, self.tasks_health_check))
        self.thread_list.append(self.thread_pool.submit(function_wrapper, self.submit_log))
        self.thread_list.append(self.thread_pool.submit(function_wrapper, self.get_new_task))
        self.thread_list.append(self.thread_pool.submit(function_wrapper, self.get_stop_task))
        self.thread_list.append(self.thread_pool.submit(
            function_wrapper, self.follow_daemon, TelemetryReporter, 'TELEMETRY_REPORTER'
        ))
        self.thread_list.append(self.thread_pool.submit(
            function_wrapper, self.follow_daemon, ImageStreamer, 'IMAGE_STREAMER'
        ))

    def wait_all(self):
        wait(self.thread_list)
