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
    title = "Hyperparameters Selector"

    def __init__(self, hyperparameters: dict):
        self.editor = Editor(
            hyperparameters, height_lines=50, language_mode="yaml", auto_format=True
        )

        # Model Benchmark
        self.run_model_benchmark_checkbox = Checkbox(
            content="Run Model Benchmark evaluation", checked=True
        )
        self.run_speedtest_checkbox = Checkbox(content="Run speed test", checked=True)

        self.model_benchmark_field = Field(
            Container(
                widgets=[
                    self.run_model_benchmark_checkbox,
                    self.run_speedtest_checkbox,
                ]
            ),
            title="Model Evaluation Benchmark",
            description=f"Generate evalutaion dashboard with visualizations and detailed analysis of the model performance after training. The best checkpoint will be used for evaluation. You can also run speed test to evaluate model inference speed.",
        )
        docs_link = '<a href="https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/" target="_blank">documentation</a>'
        self.model_benchmark_learn_more = Text(
            f"Learn more about Model Benchmark in the {docs_link}.", status="info"
        )

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container(
            [
                self.editor,
                self.model_benchmark_field,
                self.model_benchmark_learn_more,
                self.validator_text,
                self.button,
            ]
        )
        self.card = Card(
            title="Hyperparameters",
            description="Set hyperparameters for training",
            content=container,
            lock_message="Select model to unlock",
        )
        self.card.lock()

    @property
    def widgets_to_disable(self):
        return [self.editor, self.run_model_benchmark_checkbox, self.run_speedtest_checkbox]

    def set_hyperparameters(self, hyperparameters: Union[str, dict]):
        self.editor.set_text(hyperparameters)

    def get_hyperparameters(self) -> dict:
        return self.editor.get_value()

    def get_model_benchmark_checkbox_value(self) -> bool:
        return self.run_model_benchmark_checkbox.is_checked()

    def set_model_benchmark_checkbox_value(self, is_checked: bool) -> bool:
        if is_checked:
            self.run_model_benchmark_checkbox.check()
        else:
            self.run_model_benchmark_checkbox.uncheck()

    def get_speedtest_checkbox_value(self) -> bool:
        return self.run_speedtest_checkbox.is_checked()

    def set_speedtest_checkbox_value(self, is_checked: bool) -> bool:
        if is_checked:
            self.run_speedtest_checkbox.check()
        else:
            self.run_speedtest_checkbox.uncheck()

    def toggle_mb_speedtest(self, is_checked: bool):
        if is_checked:
            self.run_speedtest_checkbox.show()
        else:
            self.run_speedtest_checkbox.hide()

    def validate_step(self):
        return True