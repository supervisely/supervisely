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
    """
    :param checkpoint_name: A name for model weights, e.g., "YOLOv8n COCO (best)".
    :param model_name: The name of a model for which the weights are applicable, e.g., "YOLOv8n".
    :param architecture: Collection for a set of models, e.g., "YOLOv8".
    :param checkpoint_url: URL to download the model weights.
    :param custom_checkpoint_path: Path in Team Files to the weights.
    :param model_source: Source of the model, either "Pretrained models" or "Custom models".
    """
    checkpoint_name: str = None
    model_name: str = None
    architecture: str = None
    checkpoint_url: str = None
    custom_checkpoint_path: str = None
    model_source: str = None


@dataclass
class DeployInfo:
    checkpoint_name: str
    model_name: str
    architecture: str
    checkpoint_url: str
    custom_checkpoint_path: str
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