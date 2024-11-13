from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DoneLabel,
    Empty,
    Field,
    FolderThumbnail,
    Progress,
    ReportThumbnail,
    SlyTqdm,
    TaskLogs,
    Text,
)


class TrainingProcess:
    title = "Training Process"

    def __init__(self):
        # @TODO: add charts
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

        # self.creating_model_benchmark_report_field = Field(Empty(), "", "Creating report on model...")
        # self.creating_model_benchmark_report_field.hide()

        self.project_download_progress = Progress("Downloading datasets", hide_on_finish=True)
        self.project_download_progress.hide()

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

        self.validator_text = Text("")
        self.validator_text.hide()
        self.start_button = Button("Start")
        self.stop_button = Button("Stop", button_type="danger")

        button_container = Container(
            [self.start_button, self.stop_button, Empty()],
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

        self.task_logs = TaskLogs()
        self.task_logs.hide()

        container = Container(
            [
                self.success_message,
                self.artifacts_thumbnail,
                self.model_benchmark_report_thumbnail,
                self.model_benchmark_report_text,
                self.validator_text,
                button_container,
                self.logs_button,
                self.project_download_progress,
                self.model_download_progress_main,
                self.model_download_progress_secondary,
                self.epoch_progress,
                self.iter_progress,
                self.artifacts_upload_progress,
                self.model_benchmark_progress_main,
                self.model_benchmark_progress_secondary,
            ]
        )
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
