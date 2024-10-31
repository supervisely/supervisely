from supervisely.app.widgets import (
    Button,
    Card,
    ClassesTable,
    Container,
    Field,
    Switch,
    Text,
)
from supervisely.project.download import is_cached


class ClassesSelector:
    title = "Classes Selection"

    def __init__(self, project_id: int, classes: list):
        self.classes_table = ClassesTable(project_id=project_id)  # use dataset_ids
        self.classes_table.select_classes(classes)  # from app options

        self.remove_unlabeled_images = Switch(True)
        filter_images_without_gt_field = Field(
            self.remove_unlabeled_images,
            title="Filter images without annotations",
            description="If enabled, images without annotations depending on selected classes will be removed from training.",
        )

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container(
            [
                self.classes_table,
                filter_images_without_gt_field,
                self.validator_text,
                self.button,
            ]
        )
        self.card = Card(
            title="Training classes",
            description=(
                "Select classes that will be used for training. "
                "Supported shapes are Bitmap, Polygon, Rectangle."
            ),
            content=container,
            lock_message="Select model to unlock",
        )
        self.card.lock()

    @property
    def widgets_to_disable(self):
        return [self.classes_table, self.remove_unlabeled_images]

    def get_selected_classes(self):
        return self.classes_table.get_selected_classes()

    def set_classes(self, classes):
        self.classes_table.select_classes(classes)

    def select_all_classes(self):
        self.classes_table.select_all()

    def validate_step(self):
        self.validator_text.hide()
        selected_classes = self.classes_table.get_selected_classes()
        n_classes = len(selected_classes)
        if n_classes == 0:
            self.validator_text.set(
                text="Please select at least one class", status="error"
            )
        else:
            class_text = "class" if n_classes == 1 else "classes"
            self.validator_text.set(
                text=f"Selected {n_classes} {class_text}", status="success"
            )
        self.validator_text.show()
        return n_classes > 0
