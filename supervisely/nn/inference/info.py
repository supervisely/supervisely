from typing import NamedTuple, Dict, Any


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


class ModelInfo(NamedTuple):
    model_name: str
    architecture: str
    task_type: str
    model_source: str


class DeployInfo(NamedTuple):
    model_name: str
    architecture: str
    task_type: str
    model_source: str
    device: str
    runtime: str
    hardware: str
    deploy_params: Dict[str, Any]


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