# coding: utf-8
from packaging import version
import supervisely_lib as sly

from worker import constants
from worker.task_sly import TaskSly


class TaskPullDockerImage(TaskSly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.docker_runtime = 'runc'  # or 'nvidia'
        self._docker_api = None  # must be set by someone

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

    def task_main_func(self):
        self.logger.info('TASK_START', extra={'event_type': sly.EventType.TASK_STARTED})
        sly.docker_utils.docker_pull_if_needed(self._docker_api, self.docker_image_name, self.info['pull_policy'], self.logger)
        docker_img = self._docker_api.images.get(self.docker_image_name)
        if constants.CHECK_VERSION_COMPATIBILITY():
            self._validate_version(self.info["agent_version"], docker_img.labels.get("VERSION", None))

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
            self.logger.critical('Agent version is lower than plugin version. Please, upgrade agent.')

    def end_log_stop(self):
        return sly.EventType.TASK_STOPPED

    def end_log_crash(self, e):
        return sly.EventType.TASK_CRASHED

    def end_log_finish(self):
        return sly.EventType.TASK_FINISHED

    def report_start(self):
        pass