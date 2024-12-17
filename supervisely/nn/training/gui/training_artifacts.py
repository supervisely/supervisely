from typing import Any, Dict

from supervisely import Api
from supervisely.app.widgets import (
    Card,
    Container,
    Empty,
    Field,
    Flexbox,
    FolderThumbnail,
    ReportThumbnail,
    Text,
)

PYTORCH_ICON = "https://img.icons8.com/?size=100&id=jH4BpkMnRrU5&format=png&color=000000"
ONNX_ICON = "https://artwork.lfaidata.foundation/projects/onnx/icon/color/onnx-icon-color.png"
TRT_ICON = "https://img.icons8.com/?size=100&id=yqf95864UzeQ&format=png&color=000000"


class TrainingArtifacts:
    title = "Training Artifacts"
    description = "All outputs of the training process will appear here"
    lock_message = "Artifacts will be available after training is completed"

    def __init__(self, app_options: Dict[str, Any]):
        self.display_widgets = []
        self.success_message_text = (
            "Training completed. Training artifacts were uploaded to Team Files. "
            "You can find and open tensorboard logs in the artifacts folder via the "
            "<a href='https://ecosystem.supervisely.com/apps/tensorboard-logs-viewer' target='_blank'>Tensorboard</a> app."
        )
        self.app_options = app_options

        # GUI Components
        self.validator_text = Text("")
        self.validator_text.hide()
        self.display_widgets.extend([self.validator_text])

        # Outputs
        self.artifacts_thumbnail = FolderThumbnail()
        self.artifacts_thumbnail.hide()

        self.artifacts_field = Field(
            title="Artifacts",
            description="Contains all outputs of the training process",
            content=self.artifacts_thumbnail,
        )
        self.artifacts_field.hide()
        self.display_widgets.extend([self.artifacts_field])

        # Optional Model Benchmark
        if app_options.get("model_benchmark", False):
            self.model_benchmark_report_thumbnail = ReportThumbnail()
            self.model_benchmark_report_thumbnail.hide()

            self.model_benchmark_fail_text = Text(
                text="Model evaluation did not finish successfully. Please check the app logs for details.",
                status="error",
            )
            self.model_benchmark_fail_text.hide()

            self.model_benchmark_widgets = Container(
                [self.model_benchmark_report_thumbnail, self.model_benchmark_fail_text]
            )

            self.model_benchmark_report_field = Field(
                title="Model Benchmark",
                description="Evaluation report of the trained model",
                content=self.model_benchmark_widgets,
            )
            self.model_benchmark_report_field.hide()
            self.display_widgets.extend([self.model_benchmark_report_field])
        # -------------------------------- #

        # PyTorch, ONNX, TensorRT demo
        self.inference_demo_field = []
        model_demo = self.app_options.get("demo", None)
        if model_demo is not None:
            pytorch_demo_link = model_demo.get("pytorch", None)
            if pytorch_demo_link is not None:
                pytorch_icon = Field.Icon(image_url=PYTORCH_ICON, bg_color_rgb=[255, 255, 255])
                self.pytorch_instruction = Field(
                    title="PyTorch",
                    description="Open file",
                    description_url=pytorch_demo_link,
                    icon=pytorch_icon,
                    content=Empty(),
                )
                self.pytorch_instruction.hide()
                self.inference_demo_field.extend([self.pytorch_instruction])

            onnx_demo_link = model_demo.get("onnx", None)
            if onnx_demo_link is not None:
                if self.app_options.get("export_onnx_supported", False):
                    onnx_icon = Field.Icon(image_url=ONNX_ICON, bg_color_rgb=[255, 255, 255])
                    self.onnx_instruction = Field(
                        title="ONNX",
                        description="Open file",
                        description_url=onnx_demo_link,
                        icon=onnx_icon,
                        content=Empty(),
                    )
                    self.onnx_instruction.hide()
                    self.inference_demo_field.extend([self.onnx_instruction])

            trt_demo_link = model_demo.get("tensorrt", None)
            if trt_demo_link is not None:
                if self.app_options.get("export_tensorrt_supported", False):
                    trt_icon = Field.Icon(image_url=TRT_ICON, bg_color_rgb=[255, 255, 255])
                    self.trt_instruction = Field(
                        title="TensorRT",
                        description="Open file",
                        description_url=trt_demo_link,
                        icon=trt_icon,
                        content=Empty(),
                    )
                    self.trt_instruction.hide()
                    self.inference_demo_field.extend([self.trt_instruction])

            demo_overview_link = model_demo.get("overview", None)
            self.inference_demo_field = Field(
                title="How to run inference",
                description="Instructions on how to use your checkpoints outside of Supervisely Platform",
                content=Flexbox(self.inference_demo_field),
                title_url=demo_overview_link,
            )
            self.inference_demo_field.hide()
            self.display_widgets.extend([self.inference_demo_field])
        # -------------------------------- #

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
        return []

    def validate_step(self) -> bool:
        return True
