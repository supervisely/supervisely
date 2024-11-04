from typing import Union

from supervisely.app.widgets import Button, Card, Container, Editor, Text


class HyperparametersSelector:
    title = "Hyperparameters Selector"

    def __init__(self, hyperparameters: dict):
        self.editor = Editor(
            hyperparameters, height_lines=50, language_mode="yaml", auto_format=True
        )

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container(
            [
                self.editor,
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
        return [self.editor]

    def set_hyperparameters(self, hyperparameters: Union[str, dict]):
        self.editor.set_text(hyperparameters)

    def get_hyperparameters(self) -> dict:
        return self.editor.get_value()

    def validate_step(self):
        return True
