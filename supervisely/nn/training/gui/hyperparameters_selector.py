from typing import Union

from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Field,
    Text,
)


class HyperparametersSelector:
    title = "Hyperparameters"
    description = "Set hyperparameters for training"
    lock_message = "Select previous step to unlock"

    def __init__(self, hyperparameters: dict, app_options: dict = {}):
        # Init widgets
        self.editor = None
        self.run_model_benchmark_checkbox = None
        self.run_speedtest_checkbox = None
        self.model_benchmark_field = None
        self.model_benchmark_learn_more = None
        self.model_benchmark_auto_convert_warning = None
        self.export_onnx_checkbox = None
        self.export_tensorrt_checkbox = None
        self.export_field = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        # GUI Components
        self.editor = Editor(
            hyperparameters, height_lines=50, language_mode="yaml", auto_format=True
        )
        self.display_widgets.extend([self.editor])

        # Optional Model Benchmark
        if self.app_options.get("model_benchmark", True):
            # Model Benchmark
            self.run_model_benchmark_checkbox = Checkbox(
                content="Run Model Benchmark evaluation", checked=True
            )
            self.run_speedtest_checkbox = Checkbox(content="Run speed test", checked=False)

            self.model_benchmark_field = Field(
                title="Model Evaluation Benchmark",
                description="Generate evaluation dashboard with visualizations and detailed analysis of the model performance after training. The best checkpoint will be used for evaluation. You can also run speed test to evaluate model inference speed.",
                content=Container([self.run_model_benchmark_checkbox, self.run_speedtest_checkbox]),
            )
            docs_link = '<a href="https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/" target="_blank">documentation</a>'
            self.model_benchmark_learn_more = Text(
                f"Learn more about Model Benchmark in the {docs_link}.", status="info"
            )
            self.model_benchmark_auto_convert_warning = Text(
                text="Project will be automatically converted according to CV task and uploaded for Model Evaluation.",
                status="warning",
            )
            self.model_benchmark_auto_convert_warning.hide()

            self.display_widgets.extend(
                [
                    self.model_benchmark_field,
                    self.model_benchmark_learn_more,
                    self.model_benchmark_auto_convert_warning,
                ]
            )
        # -------------------------------- #

        # Optional Export Weights
        export_onnx_supported = self.app_options.get("export_onnx_supported", False)
        export_tensorrt_supported = self.app_options.get("export_tensorrt_supported", False)

        onnx_name = "ONNX"
        tensorrt_name = "TensorRT engine"
        export_runtimes = []
        export_runtime_names = []
        if export_onnx_supported:
            self.export_onnx_checkbox = Checkbox(content=f"Export to {onnx_name}")
            export_runtimes.append(self.export_onnx_checkbox)
            export_runtime_names.append(onnx_name)
        if export_tensorrt_supported:
            self.export_tensorrt_checkbox = Checkbox(content=f"Export to {tensorrt_name}")
            export_runtimes.append(self.export_tensorrt_checkbox)
            export_runtime_names.append(tensorrt_name)
        if export_onnx_supported or export_tensorrt_supported:
            export_field_description = ", ".join(export_runtime_names)
            runtime_container = Container(export_runtimes)
            self.export_field = Field(
                title="Export model",
                description=f"Export best checkpoint to the following formats: {export_field_description}.",
                content=runtime_container,
            )
            self.display_widgets.extend([self.export_field])
        # -------------------------------- #

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend([self.validator_text, self.button])
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
            collapsable=app_options.get("collapsable", False),
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        widgets = [self.editor]
        if self.app_options.get("model_benchmark", True):
            widgets.extend([self.run_model_benchmark_checkbox, self.run_speedtest_checkbox])
        if self.app_options.get("export_onnx_supported", False):
            widgets.append(self.export_onnx_checkbox)
        if self.app_options.get("export_tensorrt_supported", False):
            widgets.append(self.export_tensorrt_checkbox)
        return widgets

    def set_hyperparameters(self, hyperparameters: Union[str, dict]) -> None:
        self.editor.set_text(hyperparameters)

    def get_hyperparameters(self) -> dict:
        return self.editor.get_value()

    def get_model_benchmark_checkbox_value(self) -> bool:
        if self.run_model_benchmark_checkbox is not None:
            return self.run_model_benchmark_checkbox.is_checked()
        return False

    def set_model_benchmark_checkbox_value(self, is_checked: bool) -> bool:
        if self.run_model_benchmark_checkbox is not None:
            if is_checked:
                self.run_model_benchmark_checkbox.check()
            else:
                self.run_model_benchmark_checkbox.uncheck()

    def get_speedtest_checkbox_value(self) -> bool:
        if self.run_speedtest_checkbox is not None:
            return self.run_speedtest_checkbox.is_checked()
        return False

    def set_speedtest_checkbox_value(self, is_checked: bool) -> bool:
        if self.run_speedtest_checkbox is not None:
            if is_checked:
                self.run_speedtest_checkbox.check()
            else:
                self.run_speedtest_checkbox.uncheck()

    def toggle_mb_speedtest(self, is_checked: bool) -> None:
        if is_checked:
            self.run_speedtest_checkbox.show()
        else:
            self.run_speedtest_checkbox.hide()

    def get_export_onnx_checkbox_value(self) -> bool:
        if self.app_options.get("export_onnx_supported", False):
            return self.export_onnx_checkbox.is_checked()
        return False

    def set_export_onnx_checkbox_value(self, value: bool) -> None:
        if value:
            self.export_onnx_checkbox.check()
        else:
            self.export_onnx_checkbox.uncheck()

    def get_export_tensorrt_checkbox_value(self) -> bool:
        if self.app_options.get("export_tensorrt_supported", False):
            return self.export_tensorrt_checkbox.is_checked()
        return False

    def set_export_tensorrt_checkbox_value(self, value: bool) -> None:
        if value:
            self.export_tensorrt_checkbox.check()
        else:
            self.export_tensorrt_checkbox.uncheck()

    def is_export_required(self) -> bool:
        return self.get_export_onnx_checkbox_value() or self.get_export_tensorrt_checkbox_value()

    def validate_step(self) -> bool:
        return True
