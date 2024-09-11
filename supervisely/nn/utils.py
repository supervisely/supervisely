from dataclasses import dataclass


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
