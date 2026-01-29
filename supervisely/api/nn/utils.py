from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, TypedDict, Union, Unpack

import supervisely.io.env as env
from supervisely._utils import get_valid_kwargs, logger
from supervisely.api.agent_api import AgentInfo
from supervisely.api.api import Api
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts


def find_agent(api: "Api", team_id: int = None, public=True, gpu=True) -> Optional[AgentInfo]:
    """
    Find an agent in Supervisely with most available memory.

    :param team_id: Team ID. If not provided, will be taken from the current context.
    :type team_id: Optional[int]
    :param public: If True, can find a public agent.
    :type public: bool
    :param gpu: If True, find an agent with GPU.
    :type gpu: bool
    :return: Agent info
    :rtype: AgentInfo
    """
    if team_id is None:
        team_id = env.team_id()
    agents = api.agent.get_list_available(team_id, show_public=public, has_gpu=gpu)
    if len(agents) == 0:
        raise ValueError("No available agents found.")
    agent_id_memory_map = {}
    kubernetes_agents = []
    for agent in agents:
        if agent.type == "sly_agent":
            # No multi-gpu support, always take the first one
            agent_id_memory_map[agent.id] = agent.gpu_info["device_memory"][0]["available"]
        elif agent.type == "kubernetes":
            kubernetes_agents.append(agent.id)
    if len(agent_id_memory_map) > 0:
        return max(agent_id_memory_map, key=agent_id_memory_map.get)
    if len(kubernetes_agents) > 0:
        return kubernetes_agents[0]
    
def get_framework_by_path(path: str) -> Type[BaseTrainArtifacts]:
    from supervisely.nn.artifacts import (
        RITM,
        RTDETR,
        Detectron2,
        MMClassification,
        MMDetection,
        MMDetection3,
        MMPretrain,
        MMSegmentation,
        UNet,
        YOLOv5,
        YOLOv5v2,
        YOLOv8,
    )
    from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
    from supervisely.nn.utils import ModelSource

    path_obj = Path(path)
    if len(path_obj.parts) < 2:
        raise ValueError(f"Incorrect checkpoint path: '{path}'")
    parent = path_obj.parts[1]
    frameworks = {
        "/detectron2": Detectron2,
        "/mmclassification": MMClassification,
        "/mmclassification-v2": MMPretrain,
        "/mmdetection": MMDetection,
        "/mmdetection-3": MMDetection3,
        "/mmsegmentation": MMSegmentation,
        "/RITM_training": RITM,
        "/RT-DETR": RTDETR,
        "/unet": UNet,
        "/yolov5_train": YOLOv5,
        "/yolov5_2.0_train": YOLOv5v2,
        "/yolov8_train": YOLOv8,
    }
    if f"/{parent}" in frameworks:
        return frameworks[f"/{parent}"]

    
def get_artifacts_dir_and_checkpoint_name(model: str) -> Tuple[str, str]:
    if not model.startswith("/"):
        raise ValueError(f"Path must start with '/'")

    if model.startswith("/experiments"):
        if model.endswith(".pt") or model.endswith(".pth"):
            try:
                artifacts_dir, checkpoint_name = model.split("/checkpoints/")
                return artifacts_dir, checkpoint_name
            except:
                raise ValueError(
                    "Bad format of checkpoint path. Expected format: '/artifacts_dir/checkpoints/checkpoint_name'"
                )
        elif model.endswith(".onnx") or model.endswith(".engine"):
            try:
                artifacts_dir, checkpoint_name = model.split("/export/")
                return artifacts_dir, checkpoint_name
            except:
                raise ValueError(
                    "Bad format of checkpoint path. Expected format: '/artifacts_dir/export/checkpoint_name'"
                )
        else:
            raise ValueError(f"Unknown model format: '{get_file_name_with_ext(model)}'")

    framework_cls = get_framework_by_path(model)
    if framework_cls is None:
        raise ValueError(f"Unknown path: '{model}'")

    team_id = env.team_id()
    framework = framework_cls(team_id)
    checkpoint_name = get_file_name_with_ext(model)
    checkpoints_dir = model.replace(checkpoint_name, "")
    if framework.weights_folder is not None:
        artifacts_dir = checkpoints_dir.replace(framework.weights_folder, "")
    else:
        artifacts_dir = checkpoints_dir
    return artifacts_dir, checkpoint_name

def find_team_by_path(api: "Api", path: str, team_id: int = None, raise_not_found=True):
    if team_id is not None:
        if api.file.exists(team_id, path) or api.file.dir_exists(
            team_id, path, recursive=False
        ):
            return team_id
        elif raise_not_found:
            raise ValueError(f"Checkpoint '{path}' not found in team provided team")
        else:
            return None
    team_id = env.team_id(raise_not_found=False)
    if team_id is not None:
        if api.file.exists(team_id, path) or api.file.dir_exists(
            team_id, path, recursive=False
        ):
            return team_id
    teams = api.team.get_list()
    team_id = None
    for team in teams:
        if api.file.exists(team.id, path):
            if team_id is not None:
                raise ValueError("Multiple teams have the same checkpoint")
            team_id = team.id
    if team_id is None:
        if raise_not_found:
            raise ValueError("Checkpoint not found")
        else:
            return None
    return team_id

def find_apps_by_framework(api: "Api", framework: str, categories: List[str] = None):
    if categories is None:
        categories = []
    modules = api.app.get_list_ecosystem_modules(
        categories=[*categories, f"framework:{framework}"], categories_operation="and"
    )
    return modules


def run_app(
    api: "Api", agent_id: int, module_id: int, timeout: int = 100, **kwargs
):
    _attempt_delay_sec = 1
    _attempts = timeout // _attempt_delay_sec

    if kwargs.get("workspace_id") is None:
        workspace_id = env.workspace_id()
        kwargs["workspace_id"] = workspace_id
    kwargs = get_valid_kwargs(
        kwargs=kwargs,
        func=api.task.start,
        exclude=["self", "module_id", "agent_id"],
    )
    task_info = api.task.start(
        agent_id=agent_id,
        module_id=module_id,
        workspace_id=workspace_id,
        **kwargs,
    )
    ready = api.app.wait_until_ready_for_api_calls(
        task_info["id"], _attempts, _attempt_delay_sec
    )
    if not ready:
        raise TimeoutError(
            f"Task {task_info['id']} is not ready for API calls after {timeout} seconds."
        )
    return task_info
