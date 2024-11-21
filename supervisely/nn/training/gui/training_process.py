from supervisely import Api
from supervisely._utils import is_production
from supervisely.app.widgets import Card  # SelectCudaDevice,
from supervisely.app.widgets import (
    Button,
    Container,
    DoneLabel,
    Empty,
    Field,
    FolderThumbnail,
    Progress,
    ReportThumbnail,
    SelectCudaDevice,
    SlyTqdm,
    TaskLogs,
    Text,
)
from supervisely.io.env import task_id as get_task_id


class TrainingProcess:
    title = "Training Process"

    def __init__(self):
        api = Api.from_env()

        self.select_device = SelectCudaDevice()
        self.success_message = DoneLabel(
            "Training completed. Training artifacts were uploaded to Team Files."
        )
        self.success_message.hide()

        self.artifacts_thumbnail = FolderThumbnail()
        self.artifacts_thumbnail.hide()

        self.model_benchmark_report_thumbnail = ReportThumbnail()
        self.model_benchmark_report_thumbnail.hide()

        self.model_benchmark_report_text = Text(status="info", text="Creating report on model...")
        self.model_benchmark_report_text.hide()

        self.project_download_progress_main = Progress("Downloading datasets", hide_on_finish=True)
        self.project_download_progress_main.hide()

        self.project_download_progress_secondary = Progress(
            "Processing splits", hide_on_finish=True
        )
        self.project_download_progress_secondary.hide()

        self.model_download_progress_main = Progress("Downloading model files", hide_on_finish=True)
        self.model_download_progress_main.hide()

        self.model_download_progress_secondary = Progress("Downloading file", hide_on_finish=True)
        self.model_download_progress_secondary.hide()

        self.epoch_progress = Progress("Epochs")
        self.epoch_progress.hide()

        self.iter_progress = Progress("Iterations", hide_on_finish=False)
        self.iter_progress.hide()

        self.model_benchmark_progress_main = SlyTqdm()
        self.model_benchmark_progress_main.hide()

        self.model_benchmark_progress_secondary = Progress(hide_on_finish=True)
        self.model_benchmark_progress_secondary.hide()

        self.artifacts_upload_progress = Progress("Uploading artifacts", hide_on_finish=True)
        self.artifacts_upload_progress.hide()

        if is_production():
            task_id = get_task_id(raise_not_found=False)
            task_info = api.task.get_info_by_id(task_id)
            session_token = task_info["meta"]["sessionToken"]
            sly_url_prefix = f"/net/{session_token}"
            self.tensorboard_link = f"{api.server_address}{sly_url_prefix}/tensorboard"
        else:
            task_id = None
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

        # @TODO: Add GPU Selector

        button_container = Container(
            [self.start_button, self.tensorboard_button, Empty()],
            "horizontal",
            overflow="wrap",
            fractions=[1, 1, 10],
            gap=1,
        )

        self.logs_button = Button(
            text="Show logs",
            plain=True,
            button_size="mini",
            icon="zmdi zmdi-caret-down-circle",
        )

        self.task_logs = TaskLogs(task_id)
        self.task_logs.hide()
        logs_container = Container([self.logs_button, self.task_logs])

        container = Container(
            [
                self.select_device,
                self.validator_text,
                button_container,
                self.success_message,
                self.artifacts_thumbnail,
                self.model_benchmark_report_thumbnail,
                self.model_benchmark_report_text,
                self.project_download_progress_main,
                self.project_download_progress_secondary,
                self.model_download_progress_main,
                self.model_download_progress_secondary,
                self.epoch_progress,
                self.iter_progress,
                self.artifacts_upload_progress,
                self.model_benchmark_progress_main,
                self.model_benchmark_progress_secondary,
                logs_container,
            ]
        )

        # container = Container(
        #     [
        #         self.success_message,
        #         self.artifacts_thumbnail,
        #         self.model_benchmark_report_thumbnail,
        #         self.model_benchmark_report_text,
        #         self.validator_text,
        #         button_container,
        #         self.tensorboard_button,
        #         self.logs_button,
        #         self.task_logs,
        #         self.project_download_progress_main,
        #         self.project_download_progress_secondary,
        #         self.model_download_progress_main,
        #         self.model_download_progress_secondary,
        #         self.epoch_progress,
        #         self.iter_progress,
        #         self.artifacts_upload_progress,
        #         self.model_benchmark_progress_main,
        #         self.model_benchmark_progress_secondary,
        #     ]
        # )
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

    def toggle_logs(self):
        if self.task_logs.is_hidden():
            self.task_logs.show()
            self.logs_button.text = "Hide logs"
            self.logs_button.icon = "zmdi zmdi-caret-up-circle"
        else:
            self.task_logs.hide()
            self.logs_button.text = "Show logs"
            self.logs_button.icon = "zmdi zmdi-caret-down-circle"

    def get_device(self):
        return self.select_device.get_device()
