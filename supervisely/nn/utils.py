from dataclasses import dataclass

import psutil


class ModelSource:
    PRETRAINED = "Pretrained models"
    CUSTOM = "Custom models"


class RuntimeType:
    PYTORCH = "PyTorch"
    ONNXRUNTIME = "ONNXRuntime"
    TENSORRT = "TensorRT"


class ModelPrecision:
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


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
    model_precision: str
    hardware: str
    deploy_params: dict


def _get_model_name(model_info: dict):
    name = model_info.get("model_name")
    if not name:
        name = model_info.get("meta", {}).get("model_name")
    if not name:
        name = model_info.get("Model")
    if not name:
        raise ValueError("Model name not found not in model_info nor in meta.")
    return name


def get_ram_usage():
    memory = psutil.virtual_memory()
    return memory.used, memory.total


def get_gpu_usage(device: str = None):
    if device == "cpu":
        return None, None
    try:
        import torch
    except Exception as e:
        from supervisely import logger
        logger.warning(f"Cannot import torch. Install PyTorch to get GPU usage info. Error: {e}")
        return None, None
    if not torch.cuda.is_available():
        return None, None
    gpu_index = None
    if device is None or device in ["", "auto", "cuda"]:
        gpu_index = torch.cuda.current_device()
    elif isinstance(device, int):
        gpu_index = device
    elif device.startswith("cuda:"):
        try:
            gpu_index = int(device.split(":")[-1])
        except ValueError:
            return None, None
    else:
        for i in range(torch.cuda.device_count()):
            if device == torch.cuda.get_device_name(i):
                gpu_index = i
                break
        if gpu_index is None:
            return None, None
    if gpu_index is None or gpu_index > torch.cuda.device_count() or gpu_index < 0:
        return None, None
    allocated = torch.cuda.memory_allocated(gpu_index)
    total = torch.cuda.get_device_properties(gpu_index).total_memory
    return allocated, total
