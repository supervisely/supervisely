import os
from typing import Any, Dict

import supervisely.io.env as sly_env
import supervisely.nn.training.gui.utils as gui_utils
from supervisely import Api, logger
from supervisely._utils import is_production
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
from supervisely.io.fs import file_exists

PYTORCH_ICON = "https://img.icons8.com/?size=100&id=jH4BpkMnRrU5&format=png&color=000000"
ONNX_ICON = "https://artwork.lfaidata.foundation/projects/onnx/icon/color/onnx-icon-color.png"
TRT_ICON = "https://img.icons8.com/?size=100&id=yqf95864UzeQ&format=png&color=000000"

OVERVIEW_FILE_NAME = "README.md"
PYTORCH_FILE_NAME = "demo_pytorch.py"
ONNX_FILE_NAME = "demo_onnx.py"
TRT_FILE_NAME = "demo_tensorrt.py"


class TrainingArtifacts:
    title = "Training Artifacts"
    description = "All outputs of the training process will appear here"
    lock_message = "Artifacts will be available after training is completed"

    def __init__(self, api: Api, app_options: Dict[str, Any]):
        # Init widgets
        self.artifacts_thumbnail = None
        self.artifacts_field = None
        self.model_benchmark_report_thumbnail = None
        self.model_benchmark_fail_text = None
        self.model_benchmark_widgets = None
        self.model_benchmark_report_field = None
        self.pytorch_instruction = None
        self.onnx_instruction = None
        self.trt_instruction = None
        self.inference_demo_field = None
        self.validator_text = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        self.success_message_text = (
            "Training completed. Training artifacts were uploaded to Team Files. "
            "You can find and open tensorboard logs in the artifacts folder via the "
            "<a href='https://ecosystem.supervisely.com/apps/tensorboard-experiments-viewer' target='_blank'>Tensorboard Experiment Viewer</a> app."
        )

        # GUI Components
        self.validator_text = Text("")
        self.validator_text.hide()
        self.display_widgets.extend([self.validator_text])

        # Outputs
        need_generate_report = self.app_options.get("generate_report", False)
        # ------------------------------------------------------------ #

        if need_generate_report:
            self.artifacts_thumbnail = ReportThumbnail(
                title="Experiment Report",
                color="#5fa8ff",
                bg_color="#e6f3ff",
                report_type="experiment",
            )
        else:
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
        if self.app_options.get("model_benchmark", False):
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
        self.inference_demo_widgets = []

        # Demo display works only for released apps
        self.need_upload_demo = False
        model_demo = self.app_options.get("demo", None)
        if model_demo is not None:
            model_demo_path = model_demo.get("path", None)
            if model_demo_path is not None:
                model_demo_gh_link = None
                self.pytorch_instruction = None
                self.onnx_instruction = None
                self.trt_instruction = None
                if is_production():
                    task_id = sly_env.task_id()
                    task_info = api.task.get_info_by_id(task_id)
                    app_id = task_info["meta"]["app"]["id"]
                    app_info = api.app.get_info_by_id(app_id)
                    model_demo_gh_link = app_info.repo
                else:
                    app_name = sly_env.app_name()
                    module_info = gui_utils.get_module_info_by_name(api, app_name)
                    if module_info is not None:
                        model_demo_gh_link = module_info["repo"]
                    else:
                        logger.warning(
                            f"App '{app_name}' not found in Supervisely Ecosystem. Demo artifacts will not be displayed."
                        )

                if model_demo_gh_link is not None:
                    gh_branch = f"blob/{model_demo.get('branch', 'master')}"
                    link_to_demo = f"{model_demo_gh_link}/{gh_branch}/{model_demo_path}"

                    if model_demo_gh_link is not None and model_demo_path is not None:
                        # PyTorch
                        local_pytorch_demo = os.path.join(
                            os.getcwd(), model_demo_path, PYTORCH_FILE_NAME
                        )
                        if file_exists(local_pytorch_demo):
                            pytorch_demo_link = f"{link_to_demo}/{PYTORCH_FILE_NAME}"
                            pytorch_icon = Field.Icon(
                                image_url=PYTORCH_ICON, bg_color_rgb=[255, 255, 255]
                            )
                            self.pytorch_instruction = Field(
                                title="PyTorch",
                                description="Open file",
                                description_url=pytorch_demo_link,
                                icon=pytorch_icon,
                                content=Empty(),
                            )
                            self.pytorch_instruction.hide()
                            self.inference_demo_widgets.extend([self.pytorch_instruction])

                        # ONNX
                        local_onnx_demo = os.path.join(os.getcwd(), model_demo_path, ONNX_FILE_NAME)
                        if file_exists(local_onnx_demo):
                            if self.app_options.get("export_onnx_supported", False):
                                onnx_demo_link = f"{link_to_demo}/{ONNX_FILE_NAME}"
                                onnx_icon = Field.Icon(
                                    image_url=ONNX_ICON, bg_color_rgb=[255, 255, 255]
                                )
                                self.onnx_instruction = Field(
                                    title="ONNX",
                                    description="Open file",
                                    description_url=onnx_demo_link,
                                    icon=onnx_icon,
                                    content=Empty(),
                                )
                                self.onnx_instruction.hide()
                                self.inference_demo_widgets.extend([self.onnx_instruction])

                        # TensorRT
                        local_trt_demo = os.path.join(os.getcwd(), model_demo_path, TRT_FILE_NAME)
                        if file_exists(local_trt_demo):
                            if self.app_options.get("export_tensorrt_supported", False):
                                trt_demo_link = f"{link_to_demo}/{TRT_FILE_NAME}"
                                trt_icon = Field.Icon(
                                    image_url=TRT_ICON, bg_color_rgb=[255, 255, 255]
                                )
                                self.trt_instruction = Field(
                                    title="TensorRT",
                                    description="Open file",
                                    description_url=trt_demo_link,
                                    icon=trt_icon,
                                    content=Empty(),
                                )
                                self.trt_instruction.hide()
                                self.inference_demo_widgets.extend([self.trt_instruction])

                        local_demo_overview = os.path.join(
                            os.getcwd(), model_demo_path, OVERVIEW_FILE_NAME
                        )
                        if file_exists(local_demo_overview):
                            demo_overview_link = os.path.join(link_to_demo, OVERVIEW_FILE_NAME)
                        else:
                            demo_overview_link = None

                        self.inference_demo_field = Field(
                            title="How to run inference",
                            description="Instructions on how to use your checkpoints outside of Supervisely Platform",
                            content=Flexbox(self.inference_demo_widgets),
                            title_url=demo_overview_link,
                        )
                        self.inference_demo_field.hide()
                        self.display_widgets.extend([self.inference_demo_field])
                        self.need_upload_demo = True
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

    def overview_demo_exists(self, demo_path: str):
        return file_exists(os.path.join(os.getcwd(), demo_path, OVERVIEW_FILE_NAME))

    def pytorch_demo_exists(self, demo_path: str):
        return file_exists(os.path.join(os.getcwd(), demo_path, PYTORCH_FILE_NAME))

    def onnx_demo_exists(self, demo_path: str):
        return file_exists(os.path.join(os.getcwd(), demo_path, ONNX_FILE_NAME))

    def trt_demo_exists(self, demo_path: str):
        return file_exists(os.path.join(os.getcwd(), demo_path, TRT_FILE_NAME))
