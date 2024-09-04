from __future__ import annotations
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional, Union
from supervisely.app.widgets import Select, Button, Flexbox
from supervisely.sly_logger import logger


class SelectCudaDevice(Widget):
    """
    A widget for selecting a CUDA device.

    This widget allows to select a CUDA device (and optional CPU device) from a list of detected devices on the machine.
    It displays the devices along with their reserved/total RAM values.

    :param get_list_on_init: Whether to retrieve and display the list of CUDA devices upon initialization.
    :type get_list_on_init: bool, optional
    :param sort_by_free_ram: Whether to sort the CUDA devices by their available free RAM.
    :type sort_by_free_ram: bool, optional
    :param include_cpu_option: Whether to include an option to select the CPU in the device list.
    :type include_cpu_option: bool, optional
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        get_list_on_init: Optional[bool] = True,
        sort_by_free_ram: Optional[bool] = False,
        include_cpu_option: Optional[bool] = False,
        widget_id: str = None,
    ):
        self._select = Select([])
        self._refresh_button = Button(
            text="", button_type="text", button_size="large", icon="zmdi zmdi-refresh"
        )
        self._refresh_button.click(self.refresh)
        self._content = Flexbox([self._select, self._refresh_button])

        self._sort_by_free_ram = sort_by_free_ram
        self._include_cpu_option = include_cpu_option
        if get_list_on_init:
            self.refresh()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def refresh(self) -> None:
        """Refreshes the list of available CUDA devices and updates the selector's items.

        :return: None
        """
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
            logger.warn("Unable to import Torch", extra={"error message": str(ie)})
            return

        devices = {}
        cuda.init()
        if not cuda.is_available():
            logger.warn("CUDA is not available")
            return
        try:
            for idx in range(cuda.device_count()):
                current_device = f"cuda:{idx}"
                full_device_name = f"{cuda.get_device_name(idx)} ({current_device})"
                free_mem, total_mem = cuda.mem_get_info(current_device)

                convert_to_gb = lambda number: round(number / 1024**3, 1)
                right_text = (
                    f"{convert_to_gb(total_mem - free_mem)} GB / {convert_to_gb(total_mem)} GB"
                )

                devices[full_device_name] = {
                    "device_idx": current_device,
                    "right_text": right_text,
                    "free": free_mem,
                }
            if sort_by_free_ram:
                devices = dict(
                    sorted(devices.items(), key=lambda item: item[1]["free"], reverse=True)
                )
        except Exception as e:
            logger.warning(repr(e))
        finally:
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
        """Gets the currently selected device.
        This method returns the value of the currently selected device.

        :return: The value of the selected device (e.g. 'cuda:0', 'cpu', etc.), or None if no device is selected.
        :rtype: Optional[str]
        """
        return self._select.get_value()

    def set_device(self, value: str) -> None:
        """Sets the currently selected device.

        This method updates the selector with the provided device value.

        :param value: The value of the device to be selected.
        :type value: str
        :return: None
        """
        return self._select.set_value(value)
