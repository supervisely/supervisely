# coding: utf-8
from enum import Enum
from supervisely_lib.task.progress import Progress
from docker.errors import DockerException, ImageNotFound


class PullPolicy(Enum):
    def __str__(self):
        return str(self.value)

    ALWAYS = 'Always'.lower()
    IF_AVAILABLE = 'IfAvailable'.lower()
    IF_NOT_PRESENT = 'IfNotPresent'.lower()
    NEVER = 'Never'.lower()


def docker_pull_if_needed(docker_api, docker_image_name, policy, logger):
    if policy == PullPolicy.ALWAYS:
        _docker_pull(docker_api, docker_image_name, logger)
    elif policy == PullPolicy.NEVER:
        pass
    elif policy == PullPolicy.IF_NOT_PRESENT:
        if not _docker_image_exists(docker_api, docker_image_name):
            _docker_pull(docker_api, docker_image_name, logger)
    elif policy == PullPolicy.IF_AVAILABLE:
        _docker_pull(docker_api, docker_image_name, logger, raise_exception=False)
    if not _docker_image_exists(docker_api, docker_image_name):
        raise RuntimeError("Docker image not found. Agent's PULL_POLICY is {!r}".format(str(policy)))


def _docker_pull(docker_api, docker_image_name, logger, raise_exception=True):
    logger.info('Docker image will be pulled', extra={'image_name': docker_image_name})
    progress_dummy = Progress('Pulling image...', 1, ext_logger=logger)
    progress_dummy.iter_done_report()
    try:
        pulled_img = docker_api.images.pull(docker_image_name)
        logger.info('Docker image has been pulled', extra={'pulled': {'tags': pulled_img.tags, 'id': pulled_img.id}})
    except DockerException as e:
        if raise_exception is True:
            raise DockerException('Unable to pull image: see actual error above. '
                                  'Please, run the task again or contact support team.')
        else:
            logger.warn("Pulling step is skipped. Unable to pull image: {!r}.".format(str(e)))


def _docker_image_exists(docker_api, docker_image_name):
    try:
        docker_img = docker_api.images.get(docker_image_name)
    except ImageNotFound:
        return False
    return True