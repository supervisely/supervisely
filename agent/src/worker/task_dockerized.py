# coding: utf-8

from enum import Enum
from threading import Lock
import json
from docker.errors import DockerException, ImageNotFound as DockerImageNotFound
from packaging import version

import supervisely_lib as sly

from worker.agent_utils import TaskDirCleaner
from worker import constants
from worker.task_sly import TaskSly


class TaskStep(Enum):
    NOTHING = 0
    DOWNLOAD = 1
    MAIN = 2
    UPLOAD = 3


# task with main work in separate container and with sequential steps
class TaskDockerized(TaskSly):
    step_name_mapping = {
        'DOWNLOAD': TaskStep.DOWNLOAD,
        'MAIN': TaskStep.MAIN,
        'UPLOAD': TaskStep.UPLOAD,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docker_runtime = 'runc'  # or 'nvidia'
        self.entrypoint = "/workdir/src/main.py"
        self.action_map = {}

        self.completed_step = TaskStep.NOTHING
        self.task_dir_cleaner = TaskDirCleaner(self.dir_task)

        self._docker_api = None  # must be set by someone

        self._container = None
        self._container_lock = Lock()  # to drop container from different threads
        self.docker_image_name = self.info.get('docker_image', None)
        if self.docker_image_name is not None and ':' not in self.docker_image_name:
            self.docker_image_name += ':latest'
        self.docker_pulled = False  # in task

    @property
    def docker_api(self):
        return self._docker_api

    @docker_api.setter
    def docker_api(self, val):
        self._docker_api = val

    def report_step_done(self, curr_step):
        if self.completed_step.value < curr_step.value:
            self.logger.info('STEP_DONE', extra={'step': curr_step.name, 'event_type': sly.EventType.STEP_COMPLETE})
            self.completed_step = curr_step

    def task_main_func(self):
        self.task_dir_cleaner.forbid_dir_cleaning()

        self.docker_pull_if_needed()

        last_step_str = self.info.get('last_complete_step')
        self.logger.info('LAST_COMPLETE_STEP', extra={'step': last_step_str})
        self.completed_step = self.step_name_mapping.get(last_step_str, TaskStep.NOTHING)

        for curr_step, curr_method in [
            (TaskStep.DOWNLOAD, self.download_step),
            (TaskStep.MAIN,     self.main_step),
            (TaskStep.UPLOAD,   self.upload_step),
        ]:
            if self.completed_step.value < curr_step.value:
                self.logger.info('BEFORE_STEP', extra={'step': curr_step.name})
                curr_method()

        self.task_dir_cleaner.allow_cleaning()

    # don't forget to report_step_done
    def download_step(self):
        raise NotImplementedError()

    # don't forget to report_step_done
    def upload_step(self):
        raise NotImplementedError()

    def before_main_step(self):
        raise NotImplementedError()

    def main_step_envs(self):
        return {}

    def main_step(self):
        self.before_main_step()
        sly.fs.log_tree(self.dir_task, self.logger)
        self.spawn_container(add_envs=self.main_step_envs())
        self.process_logs()
        self.drop_container_and_check_status()
        self.report_step_done(TaskStep.MAIN)

    def run(self):
        try:
            super().run()
        finally:
            if self._stop_event.is_set():
                self.task_dir_cleaner.allow_cleaning()
            self._drop_container()  # if something occurred

    def clean_task_dir(self):
        self.task_dir_cleaner.clean()

    def _docker_pull(self):
        self.logger.info('Docker image will be pulled', extra={'image_name': self.docker_image_name})
        progress_dummy = sly.Progress('Pulling image...', 1, ext_logger=self.logger)
        progress_dummy.iter_done_report()
        try:
            pulled_img = self._docker_api.images.pull(self.docker_image_name)
        except DockerException:
            raise DockerException('Unable to pull image: not enough free disk space or something wrong with DockerHub.'
                                  ' Please, run the task again or email support.')
        self.logger.info('Docker image has been pulled', extra={'pulled': {'tags': pulled_img.tags, 'id': pulled_img.id}})

        if constants.CHECK_VERSION_COMPATIBILITY():
            self._validate_version(self.info["agent_version"], pulled_img.labels.get("VERSION", None))

    def _validate_version(self, agent_image, plugin_image):
        self.logger.info('Check if agent and plugin versions are compatible')

        def get_version(docker_image):
            if docker_image is None:
                return None
            image_parts = docker_image.strip().split(":")
            if len(image_parts) != 2:
                return None
            return image_parts[1]

        agent_version = get_version(agent_image)
        plugin_version = get_version(plugin_image)

        if agent_version is None:
            self.logger.info('Unknown agent version')
            return

        if plugin_version is None:
            self.logger.info('Unknown plugin version')
            return

        av = version.parse(agent_version)
        pv = version.parse(plugin_version)

        if type(av) is version.LegacyVersion or type(pv) is version.LegacyVersion:
            self.logger.info('Invalid semantic version, can not compare')
            return

        if av.release[0] < pv.release[0]:
            self.logger.critical('Agent version is lower than plugin version. Please, update agent.')

    def _docker_image_exists(self):
        try:
            _ = self._docker_api.images.get(self.docker_image_name)
        except DockerImageNotFound:
            return False
        return True

    def docker_pull_if_needed(self):
        if self.docker_pulled:
            return
        if constants.PULL_ALWAYS() or (not self._docker_image_exists()):
            self._docker_pull()
        self.docker_pulled = True

    def _get_task_volumes(self):
        return {self.dir_task_host: {'bind': '/sly_task_data', 'mode': 'rw'}}

    def spawn_container(self, add_envs=None):
        if add_envs is None:
            add_envs = {}
        self._container_lock.acquire()
        volumes = self._get_task_volumes()
        try:
            self._container = self._docker_api.containers.run(
                self.docker_image_name,
                runtime=self.docker_runtime,
                entrypoint=["sh", "-c", "python -u {}".format(self.entrypoint)],
                detach=True,
                name='sly_task_{}_{}'.format(self.info['task_id'], constants.TASKS_DOCKER_LABEL()),
                remove=False,
                volumes=volumes,
                environment={'LOG_LEVEL': 'DEBUG', 'LANG': 'C.UTF-8', **add_envs},
                labels={'ecosystem': 'supervisely',
                        'ecosystem_token': constants.TASKS_DOCKER_LABEL(),
                        'task_id': str(self.info['task_id'])},
                shm_size="1G",
                stdin_open=False,
                tty=False
            )
            self._container.reload()
            self.logger.debug('After spawning. Container status: {}'.format(str(self._container.status)))
            self.logger.info('Docker container is spawned',extra={'container_id': self._container.id, 'container_name': self._container.name})
        finally:
            self._container_lock.release()

    def _stop_wait_container(self):
        status = {}
        container = self._container  # don't lock, fail if someone will remove container
        if container is not None:
            container.stop(timeout=2)
            status = container.wait()
        return status

    def _drop_container(self):
        self._container_lock.acquire()
        try:
            if self._container is not None:
                self._container.remove(force=True)
                self._container = None
        finally:
            self._container_lock.release()

    def drop_container_and_check_status(self):
        status = self._stop_wait_container()
        if (len(status) > 0) and (status['StatusCode'] != 0):  # StatusCode must exist
            raise RuntimeError('Task container finished with non-zero status: {}'.format(str(status)))
        self.logger.debug('Task container finished with status: {}'.format(str(status)))
        self._drop_container()
        return status

    @classmethod
    def parse_log_line(cls, log_line):
        msg = ''
        lvl = 'INFO'
        try:
            jlog = json.loads(log_line)
            msg = jlog['message']
            del jlog['message']
            lvl = jlog['level'].upper()
            del jlog['level']
        except (KeyError, ValueError, TypeError):
            jlog = {'cont_msg': str(log_line)}

        if 'event_type' not in jlog:
            jlog['event_type'] = sly.EventType.LOGJ

        return msg, jlog, lvl

    def call_event_function(self, jlog):
        et = jlog['event_type']
        if et in self.action_map:
            return self.action_map[et](jlog)
        return {}

    def process_logs(self):
        logs_found = False
        for log_line in self._container.logs(stream=True):
            logs_found = True
            log_line = log_line.decode("utf-8")
            msg, res_log, lvl = self.parse_log_line(log_line)
            output = self.call_event_function(res_log)

            lvl_description = sly.LOGGING_LEVELS.get(lvl, None)
            if lvl_description is not None:
                lvl_int = lvl_description.int
            else:
                lvl_int = sly.LOGGING_LEVELS['INFO'].int

            self.logger.log(lvl_int, msg, extra={**res_log, **output})

        if not logs_found:
            self.logger.warn('No logs obtained from container.')  # check if bug occurred
