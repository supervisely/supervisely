from supervisely.app.widgets import (
    Button,
    Card,
    ClassesTable,
    Container,
    Field,
    Switch,
    Text,
)


class ClassesSelector:
    title = "Classes Selector"

    def __init__(self, project_id: int, classes: list):
        self.classes_table = ClassesTable(project_id=project_id)  # use dataset_ids
        if len(classes) > 0:
            self.classes_table.select_classes(classes)  # from app options
        else:
            self.classes_table.select_all()

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
            lock_message="Select dataset splits to unlock",
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

    def set_train_val_datasets(self, train_dataset_id: int, val_dataset_id: int):
        self.classes_table.set_dataset_ids([train_dataset_id, val_dataset_id])

    def validate_step(self):
        self.validator_text.hide()
        selected_classes = self.classes_table.get_selected_classes()
        table_data = self.classes_table._table_data

        empty_classes = [
            row[0]["data"]
            for row in table_data
            if row[0]["data"] in selected_classes and row[2]["data"] == 0 and row[3]["data"] == 0
        ]

        n_classes = len(selected_classes)
        if n_classes == 0:
            self.validator_text.set(text="Please select at least one class", status="error")
        else:
            warning_text = ""
            status = "success"
            if empty_classes:
                intersections = set(selected_classes).intersection(empty_classes)
                if intersections:
                    warning_text = (
                        f". Selected class has no annotations: {', '.join(intersections)}"
                        if len(intersections) == 1
                        else f". Selected classes have no annotations: {', '.join(intersections)}"
                    )
                    if self.remove_unlabeled_images.is_on():
                        warning_text += (
                            ". Images without annotations will be removed from training."
                        )
                    else:
                        warning_text += ". Consider removing images without annotations or enabling 'Filter images without annotations' option."

                    status = "warning"

            class_text = "class" if n_classes == 1 else "classes"
            self.validator_text.set(
                text=f"Selected {n_classes} {class_text}{warning_text}", status=status
            )
        self.validator_text.show()
        return n_classes > 0
