import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union

import supervisely.io.env as env
from supervisely.io.fs import get_file_name_with_ext
from supervisely.sly_logger import logger

if TYPE_CHECKING:
    from supervisely.api.api import Api
    from supervisely.nn.experiments import ExperimentInfo

def get_runtime(runtime: str):
    from supervisely.nn.utils import RuntimeType

    if runtime is None:
        return None
    aliases = {
        str(RuntimeType.PYTORCH): RuntimeType.PYTORCH,
        str(RuntimeType.ONNXRUNTIME): RuntimeType.ONNXRUNTIME,
        str(RuntimeType.TENSORRT): RuntimeType.TENSORRT,
        "pytorch": RuntimeType.PYTORCH,
        "torch": RuntimeType.PYTORCH,
        "pt": RuntimeType.PYTORCH,
        "onnxruntime": RuntimeType.ONNXRUNTIME,
        "onnx": RuntimeType.ONNXRUNTIME,
        "tensorrt": RuntimeType.TENSORRT,
        "trt": RuntimeType.TENSORRT,
        "engine": RuntimeType.TENSORRT,
    }
    if runtime in aliases:
        return aliases[runtime]
    runtime = aliases.get(runtime.lower(), None)
    if runtime is None:
        raise ValueError(
            f"Runtime '{runtime}' is not supported. Supported runtimes are: {', '.join(aliases.keys())}"
        )
    return runtime

def _get_artifacts_dir_and_checkpoint_name(self, model: str) -> Tuple[str, str]:
        if not model.startswith("/"):
            raise ValueError(f"Path must start with '/'")
        
        if model.startswith("/experiments"):
            try:
                artifacts_dir, checkpoint_name = model.split("/checkpoints/")
                return artifacts_dir, checkpoint_name
            except:
                raise ValueError(
                    "Bad format of checkpoint path. Expected format: '/artifacts_dir/checkpoints/checkpoint_name'"
                )
        
        framework_cls = self._get_framework_by_path(model)
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
    
def _get_framework_by_path(self, path: str):
    from supervisely.nn.artifacts import (
        RITM,
        RTDETR,
        Detectron2,
        MMClassification,
        MMDetection,
        MMDetection3,
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