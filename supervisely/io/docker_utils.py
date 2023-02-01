# coding: utf-8
from enum import Enum
import json
from supervisely.task.progress import Progress


class PullPolicy(Enum):
    def __str__(self):
        return str(self.value)

    ALWAYS = "Always".lower()
    IF_AVAILABLE = "IfAvailable".lower()
    IF_NOT_PRESENT = "IfNotPresent".lower()
    NEVER = "Never".lower()


def docker_pull_if_needed(docker_api, docker_image_name, policy, logger, progress=True):
    logger.info(
        "docker_pull_if_needed args",
        extra={
            "policy": policy,
            "type(policy)": type(policy),
            "policy == PullPolicy.ALWAYS": str(policy) == str(PullPolicy.ALWAYS),
            "policy == PullPolicy.NEVER": str(policy) == str(PullPolicy.NEVER),
            "policy == PullPolicy.IF_NOT_PRESENT": str(policy) == str(PullPolicy.IF_NOT_PRESENT),
            "policy == PullPolicy.IF_AVAILABLE": str(policy) == str(PullPolicy.IF_AVAILABLE),
        },
    )
    if str(policy) == str(PullPolicy.ALWAYS):
        if progress is False:
            _docker_pull(docker_api, docker_image_name, logger)
        else:
            _docker_pull_progress(docker_api, docker_image_name, logger)
    elif str(policy) == str(PullPolicy.NEVER):
        pass
    elif str(policy) == str(PullPolicy.IF_NOT_PRESENT):
        if not _docker_image_exists(docker_api, docker_image_name):
            if progress is False:
                _docker_pull(docker_api, docker_image_name, logger)
            else:
                _docker_pull_progress(docker_api, docker_image_name, logger)
    elif str(policy) == str(PullPolicy.IF_AVAILABLE):
        if progress is False:
            _docker_pull(docker_api, docker_image_name, logger, raise_exception=True)
        else:
            _docker_pull_progress(docker_api, docker_image_name, logger, raise_exception=True)
    else:
        raise RuntimeError(f"Unknown pull policy {str(policy)}")
    if not _docker_image_exists(docker_api, docker_image_name):
        raise RuntimeError(
            f"Docker image {docker_image_name} not found. Agent's PULL_POLICY is {str(policy)}"
        )


def _docker_pull(docker_api, docker_image_name, logger, raise_exception=True):
    from docker.errors import DockerException

    logger.info("Docker image will be pulled", extra={"image_name": docker_image_name})
    progress_dummy = Progress("Pulling image...", 1, ext_logger=logger)
    progress_dummy.iter_done_report()
    try:
        pulled_img = docker_api.images.pull(docker_image_name)
        logger.info(
            "Docker image has been pulled",
            extra={"pulled": {"tags": pulled_img.tags, "id": pulled_img.id}},
        )
    except DockerException as e:
        if raise_exception is True:
            raise DockerException(
                "Unable to pull image: see actual error above. "
                "Please, run the task again or contact support team."
            )
        else:
            logger.warn("Pulling step is skipped. Unable to pull image: {!r}.".format(str(e)))


def _docker_pull_progress(docker_api, docker_image_name, logger, raise_exception=True):
    logger.info("Docker image will be pulled", extra={"image_name": docker_image_name})
    from docker.errors import DockerException

    try:
        layers_total = {}
        layers_current = {}
        progress = Progress("Pulling dockerimage", 1, is_size=True, ext_logger=logger)
        for line in docker_api.api.pull(docker_image_name, stream=True, decode=True):
            layer_id = line.get("id", None)
            progress_details = line.get("progressDetail", {})
            if "total" in progress_details and "current" in progress_details:
                layers_total[layer_id] = progress_details["total"]
                layers_current[layer_id] = progress_details["current"]
                total = sum(layers_total.values())
                current = sum(layers_current.values())
                if total > progress.total:
                    progress.set(current, total)
                    progress.report_progress()
                elif (current - progress.current) / total > 0.01:
                    progress.set(current, total)
                    progress.report_progress()

            # print(json.dumps(line, indent=4))
        logger.info("Docker image has been pulled", extra={"image_name": docker_image_name})
    except DockerException as e:
        if raise_exception is True:
            raise e
            # raise DockerException(
            #     "Unable to pull image: see actual error above. "
            #     "Please, run the task again or contact support team."
            # )
        else:
            logger.warn("Pulling step is skipped. Unable to pull image: {!r}.".format(repr(e)))


def _docker_image_exists(docker_api, docker_image_name):
    from docker.errors import ImageNotFound

    try:
        docker_img = docker_api.images.get(docker_image_name)
    except ImageNotFound:
        return False
    return True
