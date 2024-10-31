from supervisely.app.widgets import (
    Button,
    Card,
    ClassesTable,
    Container,
    Field,
    Switch,
    Text,
    Editor,
)
from supervisely.project.download import is_cached


class HyperparametersSelector:
    title = "Hyperparameters Selector"

    def __init__(self, hyperparameters: dict):
        self.editor = Editor(
            hyperparameters, height_px=700, language_mode="yaml"
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

    def validate_step(self):
        return True
