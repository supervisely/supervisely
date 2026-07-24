import os
from typing import Any, Dict, List

# Safe optional import for torch to prevent pylint import-error when the library is absent.
if "LOGLEVEL" in os.environ:
    os.environ["LOGLEVEL"] = os.environ["LOGLEVEL"].upper()
try:

    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Empty,
    Field,
    Input,
    InputNumber,
    SelectCudaDevice,
    Text,
)


class TrainingProcess:
    """TrainApp GUI component for managing the training process."""
    title = "Training Process"
    description = "Manage training process"
    lock_message = "Select previous step to unlock"

    def __init__(self, app_options: Dict[str, Any]):
        """
        :param app_options: App options.
        :type app_options: Dict[str, Any]
        """
        # Initialize widgets to None
        self.select_device = None
        self.select_device_field = None
        self.experiment_name_input = None
        self.experiment_name_field = None
        self.start_button = None
        self.stop_button = None
        self.validator_text = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        # GUI Components
        self.is_multi_gpu = self.app_options.get("multi_gpu", False)
        if self.app_options.get("device_selector", False):
            self.select_device = SelectCudaDevice(
                sort_by_free_ram=True, multiple=self.is_multi_gpu, width_px=275
            )
            select_device_field_title = None
            select_device_field_description = None
            if self.is_multi_gpu:
                select_device_field_title = "Select CUDA devices"
                select_device_field_description = "The devices on which the model will be trained."
            else:
                select_device_field_title = "Select CUDA device"
                select_device_field_description = "The device on which the model will be trained."
            self.select_device_field = Field(
                title=select_device_field_title,
                description=select_device_field_description,
                content=self.select_device,
            )
            self.display_widgets.extend([self.select_device_field])

        self.experiment_name_input = Input("Enter experiment name")
        self.experiment_name_field = Field(
            title="Experiment name",
            description="Experiment name will be saved to experiment_info.json",
            content=self.experiment_name_input,
        )

        self.start_button = Button("Start")
        self.stop_button = Button("Stop", button_type="danger")
        self.stop_button.hide()  # @TODO: implement stop and hide stop button until training starts
        # Shown only when a previous run crashed during upload (resume-upload mode)
        self.resume_button = Button("Resume Upload")
        self.resume_button.hide()
        button_container = Container(
            [self.start_button, self.stop_button, self.resume_button, Empty()],
            "horizontal",
            overflow="wrap",
            fractions=[1, 1, 1, 10],
            gap=1,
        )

        self.validator_text = Text("")
        self.validator_text.hide()

        # DEBUG (resume-upload testing): inject upload failures to exercise recovery paths.
        self.debug_fail_async_checkbox = Checkbox("DEBUG: fail async upload (test sync fallback)")
        self.debug_fail_sync_checkbox = Checkbox("DEBUG: fail async + sync upload (test task stop)")
        self.debug_stop_after_checkbox = Checkbox(
            "DEBUG: stop after N uploaded checkpoints (test resume skip-by-hash)"
        )
        self.debug_stop_after_input = InputNumber(value=1, min=1)
        self.debug_field = Field(
            title="Debug: upload failure injection",
            description="For testing resume-upload only. Leave all unchecked for normal training.",
            content=Container(
                [
                    self.debug_fail_async_checkbox,
                    self.debug_fail_sync_checkbox,
                    self.debug_stop_after_checkbox,
                    self.debug_stop_after_input,
                ]
            ),
        )

        self.display_widgets.extend(
            [self.experiment_name_field, self.debug_field, button_container, self.validator_text]
        )

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        widgets = [self.experiment_name_input]
        if self.app_options.get("device_selector", False):
            widgets.extend([self.select_device, self.select_device_field])
        return widgets

    def validate_step(self) -> bool:
        return True

    def get_device(self) -> str:
        if self.app_options.get("device_selector", False):
            return self.select_device.get_device()
        else:
            return "cuda:0"

    def get_devices(self) -> List:
        if self.app_options.get("device_selector", False):
            return self.select_device.get_devices()
        else:
            return ["cuda:0"]

    def get_device_name(self) -> str:
        device = self.get_device()
        if isinstance(device, list):
            device = device[0]

        if torch is not None and device.startswith("cuda"):
            device_name = torch.cuda.get_device_name(device)
        else:
            device_name = "CPU"

        return device_name

    def get_device_names(self) -> List[str]:
        devices = self.get_devices()
        if torch is None:
            return ["CPU"]
        device_names = []
        for device in devices:
            if device.startswith("cuda"):
                device_name = torch.cuda.get_device_name(device)
                device_names.append(device_name)
        return device_names

    def get_experiment_name(self) -> str:
        return self.experiment_name_input.get_value()

    def set_experiment_name(self, experiment_name) -> None:
        self.experiment_name_input.set_value(experiment_name)
