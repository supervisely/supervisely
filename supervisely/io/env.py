# coding: utf-8
import os
from typing import Callable, List


RAISE_IF_NOT_FOUND = True


def flag_from_env(s):
    return s.upper() in ["TRUE", "YES", "1"]


def remap_gpu_devices(in_device_ids):
    """
    Working limitation for CUDA
    :param in_device_ids: real GPU devices indexes. e.g.: [3, 4, 7]
    :return: CUDA ordered GPU indexes, e.g.: [0, 1, 2]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, in_device_ids))
    return list(range(len(in_device_ids)))


def _int_from_env(value):
    if value is None:
        return value
    return int(value)


def _parse_from_env(
    name: str, keys: List[str], postprocess_fn: Callable, default=None, raise_not_found=False
):
    for k in keys:
        if k in os.environ:
            return postprocess_fn(os.environ[k])

    # env not found
    if raise_not_found is True:
        raise KeyError(
            f"{name} is not defined as environment variable. One of the envs has to be defined: {keys}. Learn more in developer portal: https://developer.supervise.ly/getting-started/environment-variables"
        )

    return default


def agent_id(raise_not_found=False):
    return _parse_from_env(
        name="agent_id",
        keys=["AGENT_ID"],
        postprocess_fn=_int_from_env,
        default=None,
        raise_not_found=raise_not_found,
    )


def agent_storage():
    return os.environ.get("AGENT_STORAGE")


def team_id(raise_not_found=True):
    return _parse_from_env(
        name="team_id",
        keys=["CONTEXT_TEAMID", "context.teamId"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )
