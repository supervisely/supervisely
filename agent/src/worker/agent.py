# coding: utf-8

import time
import docker
import json
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import subprocess

import supervisely_lib as sly

from worker import constants
from worker.task_factory import create_task
from worker.logs_to_rpc import add_task_handler
from worker.agent_utils import LogQueue
from worker.system_info import get_hw_info, get_self_docker_image_digest
from worker.image_streamer import ImageStreamer
from worker.telemetry_reporter import TelemetryReporter


class Agent:
    def __init__(self):
        self.logger = sly.get_task_logger('agent')
        sly.change_formatters_default_values(self.logger, 'service_type', sly.ServiceType.AGENT)
        sly.change_formatters_default_values(self.logger, 'event_type', sly.EventType.LOGJ)
        self.log_queue = LogQueue()
        add_task_handler(self.logger, self.log_queue)
        sly.add_default_logging_into_file(self.logger, constants.AGENT_LOG_DIR())

        self.logger.info('Agent comes back...')

        self.task_pool_lock = threading.Lock()
        self.task_pool = {}  # task_id -> task_manager (process_id)

        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.thread_list = []
        self.daemons_list = []

        sly.fs.clean_dir(constants.AGENT_TMP_DIR())
        self._stop_missed_containers()

        self.docker_api = docker.from_env(version='auto')
        self._docker_login()

        self.logger.info('Agent is ready to get tasks.')
        self.api = sly.AgentAPI(constants.TOKEN(), constants.SERVER_ADDRESS(), self.logger,
                                constants.TIMEOUT_CONFIG_PATH())
        self.agent_connect_initially()
        self.logger.info('Agent connected to server.')

    def agent_connect_initially(self):
        try:
            hw_info = get_hw_info()
        except Exception:
            hw_info = {}
            self.logger.debug('Hardware information can not be obtained')

        docker_inspect_cmd = "curl -s --unix-socket /var/run/docker.sock http:/containers/$(hostname)/json"
        docker_img_info = subprocess.Popen([docker_inspect_cmd], 
                                            shell=True, executable="/bin/bash", 
                                            stdout=subprocess.PIPE).communicate()[0]

        agent_info = {
            'hardware_info': hw_info,
            'agent_image': json.loads(docker_img_info)["Config"]["Image"],
            'agent_image_digest': get_self_docker_image_digest()
        }

        self.api.simple_request('AgentConnected', sly.api_proto.ServerInfo,
                                sly.api_proto.AgentInfo(info=json.dumps(agent_info)))

    def send_connect_info(self):
        while True:
            time.sleep(2)
            self.api.simple_request('AgentPing', sly.api_proto.Empty, sly.api_proto.Empty())

    def get_new_task(self):
        for task in self.api.get_endless_stream('GetNewTask', sly.api_proto.Task, sly.api_proto.Empty()):
            task_msg = json.loads(task.data)
            self.logger.debug('GET_NEW_TASK', extra={'task_msg': task_msg})
            self.start_task(task_msg)

    def get_stop_task(self):
        for task in self.api.get_endless_stream('GetStopTask', sly.api_proto.Id, sly.api_proto.Empty()):
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

                self.logger.info('TASK_MISSED', extra={'service_type': sly.ServiceType.TASK,
                                                       'event_type': sly.EventType.TASK_STOPPED,
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
        task_extra = {'event_type': sly.EventType.TASK_REMOVED,
                      'task_id': task_id,
                      'exit_status': self.task_pool[task_id],
                      'exit_code': self.task_pool[task_id].exitcode,
                      'service_type': sly.ServiceType.TASK}

        self.logger.info('REMOVE_TASK_TEMP_DATA IF NECESSARY', extra=task_extra)
        self.task_pool[task_id].clean_task_dir()

        del self.task_pool[task_id]
        self.logger.info('TASK_REMOVED', extra=task_extra)

    @staticmethod
    def _remove_containers(label_filter):
        dc = docker.from_env()
        stop_list = dc.containers.list(all=True, filters=label_filter)
        for cont in stop_list:
            cont.remove(force=True)
        return stop_list

    def _stop_missed_containers(self):
        self.logger.info('Searching for missed containers...')
        label_filter = {'label': 'ecosystem_token={}'.format(constants.TASKS_DOCKER_LABEL())}

        stopped_list = Agent._remove_containers(label_filter=label_filter)

        if len(stopped_list) == 0:
            self.logger.info('There are no missed containers.')

        for cont in stopped_list:
            self.logger.info('Container stopped', extra={'cont_id': cont.id, 'labels': cont.labels})
            self.logger.info('TASK_MISSED', extra={'service_type': sly.ServiceType.TASK,
                                                   'event_type': sly.EventType.MISSED_TASK_FOUND,
                                                   'task_id': int(cont.labels['task_id'])})

    def _docker_login(self):
        doc_logs = constants.DOCKER_LOGIN().split(',')
        doc_pasws = constants.DOCKER_PASSWORD().split(',')
        doc_regs = constants.DOCKER_REGISTRY().split(',')

        for login, password, registry in zip(doc_logs, doc_pasws, doc_regs):
            if registry:
                doc_login = self.docker_api.login(username=login, password=password, registry=registry)
                self.logger.info('DOCKER_CLIENT_LOGIN_SUCCESS', extra={**doc_login, 'registry': registry})

    def submit_log(self):
        while True:
            log_lines = self.log_queue.get_log_batch_blocking()
            self.api.simple_request('Log', sly.api_proto.Empty, sly.api_proto.LogLines(data=log_lines))

    def follow_daemon(self, process_cls, name, sleep_sec=5):
        proc = process_cls()
        self.daemons_list.append(proc)
        try:
            proc.start()
            while True:
                if not proc.is_alive():
                    err_msg = '{}_CRASHED'.format(name)
                    self.logger.error('Agent process is dead.', extra={'exc_str': err_msg})
                    time.sleep(1)  # an opportunity to send log
                    raise RuntimeError(err_msg)
                time.sleep(sleep_sec)
        except Exception as e:
            proc.terminate()
            proc.join(timeout=2)
            raise e

    def inf_loop(self):
        self.thread_list.append(self.thread_pool.submit(sly.function_wrapper, self.tasks_health_check))
        self.thread_list.append(self.thread_pool.submit(sly.function_wrapper, self.submit_log))
        self.thread_list.append(self.thread_pool.submit(sly.function_wrapper, self.get_new_task))
        self.thread_list.append(self.thread_pool.submit(sly.function_wrapper, self.get_stop_task))
        self.thread_list.append(self.thread_pool.submit(sly.function_wrapper, self.send_connect_info))
        self.thread_list.append(
            self.thread_pool.submit(sly.function_wrapper, self.follow_daemon, TelemetryReporter, 'TELEMETRY_REPORTER'))
        self.thread_list.append(
            self.thread_pool.submit(sly.function_wrapper, self.follow_daemon, ImageStreamer, 'IMAGE_STREAMER'))

    def wait_all(self):
        def terminate_all_deamons():
            for process in self.daemons_list:
                process.terminate()
                process.join(timeout=2)
                return

        futures_statuses = wait(self.thread_list, return_when='FIRST_EXCEPTION')
        for future in self.thread_list:
            if future.done():
                try:
                    future.result()
                except Exception:
                    terminate_all_deamons()
                    break

        futures_statuses = wait(self.thread_list, return_when='FIRST_EXCEPTION')
        for future in self.thread_list:
            if future.done():
                try:
                    future.result()
                except Exception:
                    terminate_all_deamons()
                    break

        if len(futures_statuses.not_done) != 0:
            raise RuntimeError("AGENT: EXCEPTION IN BASE FUTURE !!!")

