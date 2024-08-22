from dataclasses import dataclass


class RuntimeType:
    PYTORCH = "PyTorch"
    ONNXRUNTIME = "ONNXRuntime"
    TENSORRT = "TensorRT"


class ModelSource:
    PRETRAINED = "Pretrained models"
    CUSTOM = "Custom models"


@dataclass
class CheckpointInfo:
    checkpoint_name: str = None     # e.g. "YOLOv8-L (COCO)"
    architecture: str = None        # e.g. "YOLOv8"
    model_source: str = None        # e.g. "Pretrained models"


@dataclass
class DeployInfo:
    checkpoint_name: str
    architecture: str
    model_source: str
    task_type: str
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