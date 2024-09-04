from __future__ import annotations
from supervisely.app.widgets import Widget, ConditionalWidget
from typing import List, Dict, Optional, Union
from supervisely.app.widgets import Select, Button, Flexbox
from supervisely.sly_logger import logger
from supervisely.api.api import Api
from supervisely.io import env

# from supervisely.decorators.profile import timeit


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
            text="", button_type="text", icon="zmdi zmdi-refresh", plain=True, button_size="large"
        )
        self._refresh_button.click(self.refresh)
        self._content = Flexbox([self._select, self._refresh_button])

        self._sort_by_free_ram = sort_by_free_ram
        self._include_cpu_option = include_cpu_option
        self._agent_info = self._get_agent_info(Api())

        if get_list_on_init:
            self.refresh()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def refresh(self) -> None:
        """Refreshes the list of available CUDA devices and updates the selector's items.

        :return: None
        """

        cuda_devices = self._get_gpu_infos(self._sort_by_free_ram)
        items = []
        if cuda_devices is not None:
            for info in cuda_devices:
                item = Select.Item(
                    value=info["value"],
                    label=info["label"],
                    right_text=info["right_text"],
                )
                items.append(item)

        if self._include_cpu_option:
            items.append(Select.Item(value="cpu", label="CPU"))

        if len(items) == 0:
            self._select.set([Select.Item(None, "No devices found")])
            self._select.disable()
            return
        self._select.set(items)

    def _get_agent_info(self, api: Api):
        available_agents = api.agent.get_list_available(env.team_id(), True)
        current_agent_id = api.task.get_info_by_id(env.task_id())["agentId"]
        agent_info = None
        for agent in available_agents:
            if agent.id == current_agent_id:
                agent_info = agent
                break
        return agent_info

    def _get_gpu_infos(
        self, sort_by_free_ram: bool
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:

        gpu_info = self._agent_info.gpu_info
        device_count = gpu_info["device_count"]
        if device_count == 0:
            return

        devices_names = gpu_info["device_names"]
        devices_id = [f"cuda:{i}" for i in range(device_count)]
        devices_memory = gpu_info["device_memory"]

        devices = []
        convert_to_gb = lambda number: round(number / 1024**3, 1)
        for device_name, device_id, device_memory in zip(devices_names, devices_id, devices_memory):
            device_info = {
                "value": device_id,
                "label": f"{device_name} ({device_id})",
                "right_text": f"{convert_to_gb(device_memory['available'])} / {convert_to_gb(device_memory['total'])} GB",
                "free": convert_to_gb(device_memory["available"]),
            }
            devices.append(device_info)

        if sort_by_free_ram:
            devices.sort(key=lambda x: x["free"], reverse=True)

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
