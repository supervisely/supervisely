from dataclasses import dataclass


class TaskType:
    OBJECT_DETECTION = "object detection"
    INSTANCE_SEGMENTATION = "instance segmentation"
    SEMANTIC_SEGMENTATION = "semantic segmentation"
    POSE_ESTIMATION = "pose estimation"
    SALIENT_OBJECT_SEGMENTATION = "salient object segmentation"
    PROMPT_BASED_OBJECT_DETECTION = "prompt-based object detection"
    INTERACTIVE_SEGMENTATION = "interactive segmentation"
    PROMPTABLE_SEGMENTATION = "promptable segmentation"
    TRACKING = "tracking"
    OBJECT_DETECTION_3D = "object detection 3d"


class Runtime:
    PYTORCH = "PyTorch"
    ONNX = "ONNX"
    TENSORRT = "TensorRT"


class ModelSource:
    PRETRAINED = "Pretrained models"
    CUSTOM = "Custom models"


@dataclass
class ModelInfo:
    checkpoint: str = None      # e.g. "YOLOv8-L (COCO)"
    model_variant: str = None   # e.g. "YOLOv8-L"
    architecture: str = None    # e.g. "YOLOv8"


@dataclass
class DeployInfo:
    checkpoint: str
    model_variant: str
    architecture: str
    task_type: str
    model_source: str
    device: str
    runtime: str
    hardware: str
    deploy_params: dict


def get_hardware_info(idx: int = 0) -> str:
    import subprocess
    try:
        gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]).decode("utf-8").strip()
        gpu_list = gpus.split("\n")
        if idx >= len(gpu_list):
            raise ValueError(f"No GPU found at index {idx}")
        return gpu_list[idx]
    except subprocess.CalledProcessError:
        return "CPU"