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

        self.artifacts_thumbnail = FolderThumbnail()
        # self.artifacts_thumbnail.hide()

        self.artifacts_field = Field(
            title="Artifacts",
            description="Contains all outputs of the training process",
            content=self.artifacts_thumbnail,
        )
        self.artifacts_field.hide()

        self.display_widgets.extend(
            [
                self.validator_text,
                self.artifacts_field,
            ]
        )
        # -------------------------------- #

        # Optional Model Benchmark
        if app_options.get("model_benchmark", False):
            self.model_benchmark_report_thumbnail = ReportThumbnail()
            self.model_benchmark_report_thumbnail.hide()

            self.mb_report_field = Field(
                title="Model Benchmark",
                description="Evaluation report of the trained model",
                content=self.model_benchmark_report_thumbnail,
            )
            self.mb_report_field.hide()

            self.display_widgets.extend([self.mb_report_field])
        # -------------------------------- #

        # Run inference outside of Supervisely
        self.inference_instruction_field = []

        pytorch_icon_link = (
            "https://img.icons8.com/?size=100&id=jH4BpkMnRrU5&format=png&color=000000"
        )
        pytorch_icon = Field.Icon(image_url=pytorch_icon_link, bg_color_rgb=[255, 255, 255])

        onnx_icon_link = (
            "https://artwork.lfaidata.foundation/projects/onnx/icon/color/onnx-icon-color.png"
        )
        onnx_icon = Field.Icon(image_url=onnx_icon_link, bg_color_rgb=[255, 255, 255])

        trt_icon_link = "https://img.icons8.com/?size=100&id=yqf95864UzeQ&format=png&color=000000"
        trt_icon = Field.Icon(image_url=trt_icon_link, bg_color_rgb=[255, 255, 255])

        pytorch_link = "https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_torch.py"
        self.pytorch_instruction = Field(
            title="PyTorch",
            description="Open file",
            description_url=pytorch_link,
            icon=pytorch_icon,
            content=Empty(),
        )
        self.pytorch_instruction.hide()
        self.inference_instruction_field.extend([self.pytorch_instruction])

        if self.app_options.get("export_onnx_supported", False):
            onnx_link = "https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_onnx.py"
            self.onnx_instruction = Field(
                title="ONNX",
                description="Open file",
                description_url=onnx_link,
                icon=onnx_icon,
                content=Empty(),
            )
            self.onnx_instruction.hide()
            self.inference_instruction_field.extend([self.onnx_instruction])

        if self.app_options.get("export_tensorrt_supported", False):
            trt_link = "https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/demo_trt.py"
            self.trt_instruction = Field(
                title="TensorRT",
                description="Open file",
                description_url=trt_link,
                icon=trt_icon,
                content=Empty(),
            )
            self.trt_instruction.hide()
            self.inference_instruction_field.extend([self.trt_instruction])

        self.inference_instruction_field = Field(
            title="How to run inference",
            description="Instructions on how to use your checkpoints outside of Supervisely Platform",
            content=Flexbox(self.inference_instruction_field),
            title_url="https://github.com/supervisely-ecosystem/RT-DETRv2/blob/main/supervisely_integration/demo/README.md",
        )
        self.inference_instruction_field.hide()
        self.display_widgets.extend([self.inference_instruction_field])
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
