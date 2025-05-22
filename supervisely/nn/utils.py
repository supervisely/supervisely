import subprocess
import threading
import time
from dataclasses import dataclass
from functools import wraps

import psutil

from supervisely._utils import logger as sly_logger
from supervisely.io import env


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
    return {"used": memory.used, "total": memory.total}


def get_nvidia_smi_usage():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        encoding="utf-8",
    )
    values = result.strip().split(", ")
    utilization, memory_used = values
    utilization = int(utilization)
    memory_used = float(memory_used) * 1024 * 1024
    return {"utilization": utilization, "memory_used": memory_used}


def get_gpu_usage(device: str = None):
    if device == "cpu":
        return None

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    gpu_index = None
    if device is None or device in ["", "auto", "cuda"]:
        gpu_index = torch.cuda.current_device()
    elif isinstance(device, int):
        gpu_index = device
    elif device.startswith("cuda:"):
        gpu_index = int(device.split(":")[-1])
    else:
        for i in range(torch.cuda.device_count()):
            if device == torch.cuda.get_device_name(i):
                gpu_index = i
                break
        if gpu_index is None:
            raise RuntimeError(
                f"Device '{device}' not found. Please check your device name or index."
            )
    if gpu_index is None or gpu_index > torch.cuda.device_count() or gpu_index < 0:
        raise RuntimeError(f"Invalid GPU index '{gpu_index}'. Please check your device index.")
    allocated = torch.cuda.memory_allocated(gpu_index)
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    cached = reserved - allocated
    total = torch.cuda.get_device_properties(gpu_index).total_memory

    return {
        "allocated": allocated,
        "peak": peak,
        "reserved": reserved,
        "cached": cached,
        "total": total,
    }


class GpuUsageMonitor:
    def __ini__(self, interval=None, reset_peak_memory=False, logger=None):
        try:
            import torch
        except ImportError:
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")

        if interval is None:
            interval = env.gpu_monitoring_interval()
        self._interval = interval
        self._reset_peak_memory = reset_peak_memory
        if logger is None:
            logger = sly_logger
        self._logger = logger
        self._max_errors = 3
        self._stop_event = threading.Event()
        self._thread = None

    def _loop(self):
        import torch

        while not self._stop_event.is_set():
            try:
                if self._reset_peak_memory:
                    torch.cuda.reset_peak_memory_stats()

                gpu_usage = get_gpu_usage()
                allocated = gpu_usage["allocated"] / (1024**2)
                peak = gpu_usage["peak"] / (1024**2)
                reserved = gpu_usage["reserved"] / (1024**2)
                cached = gpu_usage["cached"] / (1024**2)
                total = gpu_usage["total"] / (1024**2)
                nvidia_usage = get_nvidia_smi_usage()
                utilization = nvidia_usage["utilization"]
                memory_used = nvidia_usage["memory_used"] / (1024**2)

                self._logger.info(
                    "GPU monitoring:",
                    extra={
                        "allocated memory (MB)": round(allocated, 2),
                        "peak memory (MB)": round(peak, 2),
                        "reserved memory (MB)": round(reserved, 2),
                        "cached memory (MB)": round(cached, 2),
                        "total memory (MB)": round(total, 2),
                        "utilization (%)": utilization,
                        "system used memory (MB)": round(memory_used, 2),
                    },
                )
                self._error_count = 0
                time.sleep(self._interval)
            except Exception as e:
                self._error_count += 1
                self._logger.error(f"Error in VRAM monitoring: {str(e)}", exc_info=True)
                if self._error_count >= self._max_errors:
                    self._logger.debug(
                        f"Stopping VRAM monitoring after {self._max_errors} consecutive errors"
                    )
                    return
                time.sleep(self._interval)

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def stop(self, wait=True, timeout=None):
        if self._thread is None:
            return
        self._stop_event.set()
        if wait:
            self._thread.join(timeout=timeout)


def log_gpu_usage(func):
    try:
        import torch
    except ImportError:
        return func

    if not env.enable_gpu_monitoring():
        return func

    logger = sly_logger

    @wraps
    def wrapper(*args, **kwargs):
        try:
            gpu_usage = get_gpu_usage()
            allocated = gpu_usage["allocated"] / (1024**2)
            peak = gpu_usage["peak"] / (1024**2)
            reserved = gpu_usage["reserved"] / (1024**2)
            cached = gpu_usage["cached"] / (1024**2)
            total = gpu_usage["total"] / (1024**2)
            nvidia_usage = get_nvidia_smi_usage()
            utilization = nvidia_usage["utilization"]
            memory_used = nvidia_usage["memory_used"] / (1024**2)
            logger.info(
                "GPU monitoring:",
                extra={
                    "function": func.__name__,
                    "allocated memory (MB)": round(allocated, 2),
                    "peak memory (MB)": round(peak, 2),
                    "reserved memory (MB)": round(reserved, 2),
                    "cached memory (MB)": round(cached, 2),
                    "total memory (MB)": round(total, 2),
                    "utilization (%)": utilization,
                    "system used memory (MB)": round(memory_used, 2),
                },
            )
        except Exception as e:
            logger.error(f"Error in VRAM monitoring: {str(e)}", exc_info=True)
        return func(*args, **kwargs)

    return wrapper
