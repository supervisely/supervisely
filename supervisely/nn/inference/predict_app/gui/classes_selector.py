from typing import Dict, Any
from supervisely.app.widgets import Button, Card, ClassesTable, Container, Text


class ClassesSelector:
    title = "Classes Selector"
    description = "Select classes that will be used for inference"
    lock_message = "Select previous step to unlock"

    def __init__(self):
        # Init Step
        self.display_widgets = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.classes_table = None
        # -------------------------------- #

        # Classes
        self.classes_table = ClassesTable()
        self.classes_table.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.classes_table])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend([self.validator_text, self.button])
        # -------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()
        # -------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [self.classes_table]

    def load_from_json(self, data: Dict[str, Any]) -> None:
        if "classes" in data:
            self.set_classes(data["classes"])

    def get_selected_classes(self) -> list:
        return self.classes_table.get_selected_classes()

    def set_classes(self, classes) -> None:
        self.classes_table.select_classes(classes)

    def select_all_classes(self) -> None:
        self.classes_table.select_all()

    def get_settings(self) -> Dict[str, Any]:
        return {"classes": self.get_selected_classes()}

    def validate_step(self) -> bool:
        if self.classes_table.is_hidden():
            return True

        self.validator_text.hide()
        selected_classes = self.classes_table.get_selected_classes()
        n_classes = len(selected_classes)

        if n_classes == 0:
            self.validator_text.set(text="Please select at least one class", status="error")
            self.validator_text.show()
            return False

        class_word = "class" if n_classes == 1 else "classes"
        message_parts = [f"Selected {n_classes} {class_word}"]
        status = "success"
        is_valid = True

        self.validator_text.set(text=". ".join(message_parts), status=status)
        self.validator_text.show()
        return is_valid
