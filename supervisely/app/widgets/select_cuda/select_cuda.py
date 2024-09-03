from __future__ import annotations
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional, Union
from supervisely.app.widgets import Select, Button, Flexbox
from supervisely.sly_logger import logger

# from supervisely.decorators.profile import timeit


class SelectCudaDevice(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        get_list_on_init: bool = True,
        sort_by_free_ram: Optional[bool] = False,
        include_cpu_option: Optional[bool] = False,
        widget_id: str = None,
    ):
        self._select = Select([])
        self._refresh_button = Button(
            text="", button_type="text", icon="zmdi zmdi-refresh", plain=True, button_size="large"
        )
        self._refresh_button.click(self.refresh)
        self._content = Flexbox([self._select, self._refresh_button])

        self._sort_by_free_ram = sort_by_free_ram
        self._include_cpu_option = include_cpu_option
        if get_list_on_init:
            self.refresh()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def refresh(self) -> None:
        cuda_devices = self._get_gpu_infos(self._sort_by_free_ram)
        if cuda_devices is None:
            return
        items = [
            Select.Item(
                value=info["device_idx"],
                label=device,
                right_text=info["right_text"],
            )
            for device, info in cuda_devices.items()
        ]

        if self._include_cpu_option:
            items.append(Select.Item(value="cpu", label="CPU"))
        self._select.set(items)

    def _get_gpu_infos(
        self, sort_by_free_ram: bool
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:
        try:
            from torch import cuda
        except ImportError as ie:
            logger.warn(
                "Unable to import Torch. Run 'pip install...'", extra={"error message": str(ie)}
            )
            return

        devices = {}
        cuda.init()
        try:
            if cuda.is_available() is True:
                for idx in range(cuda.device_count()):
                    device_name = cuda.get_device_name(idx)
                    device_idx = f"cuda:{idx}"
                    try:
                        device_props = cuda.get_device_properties(idx)
                        total_mem = device_props.total_memory
                        reserved_mem = cuda.memory_reserved(idx)
                        free_mem = total_mem - reserved_mem

                        convert_to_gb = lambda number: round(number / 1024**3, 1)
                        right_text = (
                            f"{convert_to_gb(reserved_mem)} GB / {convert_to_gb(total_mem)} GB"
                        )
                        full_device_name = f"{device_name} ({device_idx})"
                        devices[full_device_name] = {
                            "device_idx": device_idx,
                            "right_text": right_text,
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

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def value_changed(self, func):
        route_path = self.get_route_path(SelectCudaDevice.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        async def _click():
            res = self._select.get_value()
            func(res)

        return _click

    def get_device(self) -> Optional[str]:
        return self._select.get_value()

    def set_device(self, value: str) -> None:
        return self._select.set_value(value)
