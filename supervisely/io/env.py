# coding: utf-8
import os
from re import L
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


def agent_id(raise_not_found=True):
    return _parse_from_env(
        name="agent_id",
        keys=["AGENT_ID"],
        postprocess_fn=_int_from_env,
        default=None,
        raise_not_found=raise_not_found,
    )


def agent_storage(raise_not_found=True):
    return _parse_from_env(
        name="agent_storage",
        keys=["AGENT_STORAGE"],
        postprocess_fn=lambda x: x,
        default=None,
        raise_not_found=raise_not_found,
    )


def team_id(raise_not_found=True):
    return _parse_from_env(
        name="team_id",
        keys=["CONTEXT_TEAMID", "context.teamId", "TEAM_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def workspace_id(raise_not_found=True):
    return _parse_from_env(
        name="workspace_id",
        keys=["CONTEXT_WORKSPACEID", "context.workspaceId", "WORKSPACE_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def project_id(raise_not_found=True):
    return _parse_from_env(
        name="project_id",
        keys=["CONTEXT_PROJECTID", "context.projectId", "modal.state.slyProjectId", "PROJECT_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def dataset_id(raise_not_found=True):
    return _parse_from_env(
        name="dataset_id",
        keys=["CONTEXT_DATASETID", "context.datasetId", "modal.state.slyDatasetId", "DATASET_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def team_files_folder(raise_not_found=True):
    return _parse_from_env(
        name="team_files_folder",
        keys=["CONTEXT_SLYFOLDER", "context.slyFolder", "modal.state.slyFolder", "FOLDER"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def folder(raise_not_found=True):
    return team_files_folder(raise_not_found)


def team_files_file(raise_not_found=True):
    return _parse_from_env(
        name="team_files_file",
        keys=["CONTEXT_SLYFILE", "context.slyFile", "modal.state.slyFile", "FILE"],
        postprocess_fn=lambda x: str(x),
        default=None,
        raise_not_found=raise_not_found,
    )


def file(raise_not_found=True):
    return team_files_file(raise_not_found)


def task_id(raise_not_found=True):
    return _parse_from_env(
        name="task_id",
        keys=["TASK_ID"],
        postprocess_fn=lambda x: int(x),
        default=None,
        raise_not_found=raise_not_found,
    )
