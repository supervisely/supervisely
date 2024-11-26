from typing import Any, Dict

from supervisely import Api
from supervisely._utils import is_production
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DoneLabel,
    Empty,
    FolderThumbnail,
    Progress,
    ReportThumbnail,
    SelectCudaDevice,
    TaskLogs,
    Text,
)
from supervisely.io.env import task_id as get_task_id


class TrainingProcess:
    title = "Training Process"

    def __init__(self, app_options: Dict[str, Any]):
        api = Api.from_env()
        self.app_options = app_options

        self.success_message = DoneLabel(
            text=(
                "Training completed. Training artifacts were uploaded to Team Files. "
                "You can find and open tensorboard logs in the artifacts folder via the "
                "<a href='https://ecosystem.supervisely.com/apps/tensorboard-logs-viewer' target='_blank'>Tensorboard</a> app."
            )
        )
        self.success_message.hide()

        self.artifacts_thumbnail = FolderThumbnail()
        self.artifacts_thumbnail.hide()

        self.model_benchmark_report_thumbnail = ReportThumbnail()
        self.model_benchmark_report_thumbnail.hide()

        self.model_benchmark_report_text = Text(status="info", text="Creating report on model...")
        self.model_benchmark_report_text.hide()

        self.progress_bar_main = Progress(hide_on_finish=False)
        self.progress_bar_main.hide()

        self.progress_bar_secondary = Progress(hide_on_finish=False)
        self.progress_bar_secondary.hide()

        if is_production():
            task_id = get_task_id(raise_not_found=False)
        else:
            task_id = None

        # Tensorboard button
        if is_production():
            task_info = api.task.get_info_by_id(task_id)
            session_token = task_info["meta"]["sessionToken"]
            sly_url_prefix = f"/net/{session_token}"
            self.tensorboard_link = f"{api.server_address}{sly_url_prefix}/tensorboard/"
        else:
            self.tensorboard_link = "http://localhost:8000/tensorboard"
        self.tensorboard_button = Button(
            "Open Tensorboard",
            button_type="info",
            plain=True,
            icon="zmdi zmdi-chart",
            link=self.tensorboard_link,
        )
        self.tensorboard_button.disable()

        self.validator_text = Text("")
        self.validator_text.hide()
        self.start_button = Button("Start")
        self.stop_button = Button("Stop", button_type="danger")
        self.stop_button.hide()  # @TODO: implement stop and hide stop button until training starts

        button_container = Container(
            [self.start_button, self.tensorboard_button, Empty()],
            "horizontal",
            overflow="wrap",
            fractions=[1, 1, 10],
            gap=1,
        )

        container_widgets = [
            self.validator_text,
            button_container,
            self.success_message,
            self.artifacts_thumbnail,
            self.model_benchmark_report_thumbnail,
            self.model_benchmark_report_text,
            self.progress_bar_main,
            self.progress_bar_secondary,
        ]

        if self.app_options.get("device_selector", False):
            self.select_device = SelectCudaDevice()
            container_widgets.insert(1, self.select_device)

        # if is_production(): # Uncomment later
        if app_options.get("show_logs_in_gui", False):
            self.logs_button = Button(
                text="Show logs",
                plain=True,
                button_size="mini",
                icon="zmdi zmdi-caret-down-circle",
            )
            self.task_logs = TaskLogs(task_id)
            self.task_logs.hide()
            logs_container = Container([self.logs_button, self.task_logs])
            container_widgets.insert(6, logs_container)

        container = Container(container_widgets)

        self.card = Card(
            title="Training Process",
            description="Track progress and manage training",
            content=container,
            lock_message="Select hyperparametrs to unlock",
        )
        self.card.lock()

    @property
    def widgets_to_disable(self):
        return []

    def validate_step(self):
        return True

    def get_device(self):
        if self.app_options.get("device_selector", False):
            return self.select_device.get_device()
        else:
            return "cuda:0"  # @TODO: implement default device

    def toggle_logs(self):
        if self.task_logs.is_hidden():
            self.task_logs.show()
            self.logs_button.text = "Hide logs"
            self.logs_button.icon = "zmdi zmdi-caret-up-circle"
        else:
            self.task_logs.hide()
            self.logs_button.text = "Show logs"
            self.logs_button.icon = "zmdi zmdi-caret-down-circle"
