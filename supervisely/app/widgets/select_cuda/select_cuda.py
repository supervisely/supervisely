from __future__ import annotations
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional
from supervisely.app.widgets.select.select import Select

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely.sly_logger import logger

try:
    from torch import cuda
except ImportError as ie:
    logger.warn("Unable to import Torch.", extra={"error message": str(ie)})


class SelectCudaDevice(Select):
    def __init__(
        self,
        sort_by_free_ram: Optional[bool] = False,
        include_cpu_option: Optional[bool] = False,
        filterable: Optional[bool] = False,
        placeholder: Optional[str] = "Select CUDA device",
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: Optional[bool] = False,
        widget_id: Optional[str] = None,
        width_percent: Optional[int] = None,
    ):
        cuda_devices = self._get_gpu_infos(sort_by_free_ram)
        items = [
            Select.Item(
                value=mem["device_idx"],
                label=device,
                right_text=f"{round(mem['reserved']/1024**3, 1)} GB / {round(mem['total']/1024**3, 1)} GB",
            )
            for device, mem in cuda_devices.items()
        ]
        if include_cpu_option:
            items.append(Select.Item(value="cpu", label="CPU"))

        super().__init__(
            items=items,
            filterable=filterable,
            placeholder=placeholder,
            multiple=multiple,
            size=size,
            widget_id=widget_id,
            width_percent=width_percent,
        )

    @staticmethod
    def _get_gpu_infos(sort_by_free_ram):
        cuda.init()
        devices = None
        try:
            if cuda.is_available() is True:
                devices = {}
                for idx in range(cuda.device_count()):
                    device_name = cuda.get_device_name(idx)
                    device_idx = f"cuda:{idx}"
                    try:
                        device_props = cuda.get_device_properties(idx)
                        total_mem = device_props.total_memory
                        reserved_mem = cuda.memory_reserved(idx)
                        free_mem = total_mem - reserved_mem

                        device_key = f"{device_name} ({device_idx})"
                        devices[device_key] = {
                            "device_idx": device_idx,
                            "total": total_mem,
                            "reserved": reserved_mem,
                            "free": free_mem,
                        }
                    except Exception as e:
                        logger.debug(repr(e))

                if sort_by_free_ram:
                    devices = dict(
                        sorted(devices.items(), key=lambda item: item[1]["free"], reverse=True)
                    )

        except Exception as e:
            logger.warning(repr(e))
        return devices
