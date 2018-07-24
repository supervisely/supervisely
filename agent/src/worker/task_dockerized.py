# coding: utf-8

from enum import Enum

from docker.errors import ImageNotFound as DockerImageNotFound
import supervisely_lib as sly
from supervisely_lib import EventType, json_loads

from .agent_utils import TaskDirCleaner
from . import constants
from .task_sly import TaskSly


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
        self.action_map = {}

        self.completed_step = TaskStep.NOTHING
        self.task_dir_cleaner = TaskDirCleaner(self.dir_task)

        self._docker_api = None  # must be set by someone

        # pre-init for static analysis
        self.container = None
        self.container_status = {}
        self.docker_image_name = self.info['docker_image']
        self.docker_pulled = False  # in task

    @property
    def docker_api(self):
        return self._docker_api

    @docker_api.setter
    def docker_api(self, val):
        self._docker_api = val

    def report_step_done(self, curr_step):
        if self.completed_step.value < curr_step.value:
            self.logger.info('STEP_DONE', extra={'step': curr_step.name, 'event_type': EventType.STEP_COMPLETE})
            self.completed_step = curr_step

    def task_main_func(self):
        self.task_dir_cleaner.forbid_dir_cleaning()

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

    def main_step(self):
        self.docker_pull_if_needed()
        self.before_main_step()
        self.spawn_container()
        self.process_logs()
        self.drop_container_and_check()
        self.report_step_done(TaskStep.MAIN)

    def run(self):
        try:
            super().run()
        finally:
            self.drop_container()  # if something occurred

    def clean_task_dir(self):
        self.task_dir_cleaner.clean()

    def _docker_pull(self):
        self.logger.info('Docker image will be pulled', extra={'image_name': self.docker_image_name})
        _ = sly.ProgressCounter('Pulling image...', 1, ext_logger=self.logger)
        pulled_img = self._docker_api.images.pull(self.docker_image_name)
        self.logger.info('Docker image has been pulled',
                         extra={'pulled': {'tags': pulled_img.tags, 'id': pulled_img.id}})

    def _docker_image_exists(self):
        try:
            _ = self._docker_api.images.get(self.docker_image_name)
        except DockerImageNotFound:
            return False
        return True

    def docker_pull_if_needed(self):
        if self.docker_pulled:
            return
        if constants.PULL_ALWAYS or (not self._docker_image_exists()):
            self._docker_pull()
        self.docker_pulled = True

    def spawn_container(self, add_envs=None):
        if add_envs is None:
            add_envs = {}
        self.container = self._docker_api.containers.run(
            self.docker_image_name,
            runtime=self.docker_runtime,
            detach=True,
            name='sly_task_{}_'.format(self.info['task_id'], sly.generate_random_string(5)),
            remove=False,
            volumes={self.dir_task_host: {'bind': '/sly_task_data',
                                          'mode': 'rw'}},
            environment={'LOG_LEVEL': 'DEBUG', **add_envs},
            labels={'ecosystem': 'supervisely',
                    'ecosystem_token': constants.TASKS_DOCKER_LABEL,
                    'task_id': str(self.info['task_id'])},
            shm_size="1G",
            stdin_open=False,
            tty=False
        )
        self.logger.info('Docker container spawned',
                         extra={'container_id': self.container.id, 'container_name': self.container.name})

    def get_container_status(self):
        if self.container is not None:
            self.container_status = self.container.wait()
        return self.container_status

    def drop_container(self):
        status = {}
        if self.container is not None:
            self.container.stop(timeout=2)
            status = self.get_container_status()
            self.container.remove(force=True)
            self.container = None
        return status

    def drop_container_and_check(self):
        status = self.drop_container()
        if (len(status) > 0) and (status['StatusCode'] != 0):  # StatusCode must exist
            raise RuntimeError('Task container finished with non-zero status: {}'.format(str(status)))
        return status

    @classmethod
    def parse_log_line(cls, log_line):
        msg = ''
        try:
            jlog = json_loads(log_line)
            msg = jlog['message']
            del jlog['message']
        except (KeyError, ValueError, TypeError):
            jlog = {'cont_msg': str(log_line), 'event_type': EventType.LOGJ}

        if 'event_type' not in jlog:
            jlog['event_type'] = EventType.LOGJ
        return msg, jlog

    def call_event_function(self, jlog):
        et = jlog['event_type']
        if et in self.action_map:
            return self.action_map[et](jlog)
        return {}

    def process_logs(self):
        for log_line in self.container.logs(stream=True):
            log_line = log_line.decode("utf-8")
            msg, res_log = self.parse_log_line(log_line)
            output = self.call_event_function(res_log)
            self.logger.info(msg, extra={**res_log, **output})
