from typing import Any, Dict

from supervisely import Api
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DoneLabel,
    Empty,
    Field,
    FolderThumbnail,
    Input,
    ReportThumbnail,
    SelectCudaDevice,
    Text,
)


class TrainingProcess:
    title = "Training Process"
    description = "Manage training process"
    lock_message = "Select hyperparametrs to unlock"

    def __init__(self, app_options: Dict[str, Any]):
        self.app_options = app_options
        self.experiment_name_input = Input("Enter experiment name")
        self.experiment_name_field = Field(
            title="Experiment name",
            description="Experiment name will be saved to experiment_info.json",
            content=self.experiment_name_input,
        )

        self.success_message_text = (
            "Training completed. Training artifacts were uploaded to Team Files. "
            "You can find and open tensorboard logs in the artifacts folder via the "
            "<a href='https://ecosystem.supervisely.com/apps/tensorboard-logs-viewer' target='_blank'>Tensorboard</a> app."
        )
        self.success_message = DoneLabel(text=self.success_message_text)
        self.success_message.hide()

        self.artifacts_thumbnail = FolderThumbnail()
        self.artifacts_thumbnail.hide()

        self.model_benchmark_report_thumbnail = ReportThumbnail()
        self.model_benchmark_report_thumbnail.hide()

        self.validator_text = Text("")
        self.validator_text.hide()
        self.start_button = Button("Start")
        self.stop_button = Button("Stop", button_type="danger")
        self.stop_button.hide()  # @TODO: implement stop and hide stop button until training starts

        button_container = Container(
            [self.start_button, self.stop_button, Empty()],
            "horizontal",
            overflow="wrap",
            fractions=[1, 1, 10],
            gap=1,
        )

        container_widgets = [
            self.experiment_name_field,
            button_container,
            self.validator_text,
            self.artifacts_thumbnail,
            self.model_benchmark_report_thumbnail,
        ]

        if self.app_options.get("device_selector", False):
            self.select_device = SelectCudaDevice()
            self.select_cuda_device_field = Field(
                title="Select CUDA device",
                description="The device on which the model will be trained",
                content=self.select_device,
            )
            container_widgets.insert(1, self.select_cuda_device_field)

        container = Container(container_widgets)

        self.card = Card(
            title=self.title,
            description=self.description,
            content=container,
            lock_message=self.lock_message,
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        widgets = [self.experiment_name_input]
        if self.app_options.get("device_selector", False):
            widgets.append(self.experiment_name_input)
        return widgets

        return []

    def validate_step(self) -> bool:
        return True

    def get_device(self) -> str:
        if self.app_options.get("device_selector", False):
            return self.select_device.get_device()
        else:
            return "cuda:0"

    def get_experiment_name(self) -> str:
        return self.experiment_name_input.get_value()

    def set_experiment_name(self, experiment_name) -> None:
        self.experiment_name_input.set_value(experiment_name)
