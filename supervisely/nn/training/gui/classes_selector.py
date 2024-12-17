from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.app.widgets import Button, Card, ClassesTable, Container, Text


class ClassesSelector:
    title = "Classes Selector"
    description = (
        "Select classes that will be used for training. "
        "Supported shapes are Bitmap, Polygon, Rectangle."
    )
    lock_message = "Select training and validation splits to unlock"

    def __init__(self, project_id: int, classes: list, app_options: dict = {}):
        self.display_widgets = []

        # GUI Components
        if is_development() or is_debug_with_sly_net():
            qa_stats_link = abs_url(f"projects/{project_id}/stats/datasets")
        else:
            qa_stats_link = f"/projects/{project_id}/stats/datasets"
        qa_stats_text = Text(
            text=f"<i class='zmdi zmdi-chart-donut' style='color: #7f858e'></i> <a href='{qa_stats_link}' target='_blank'> <b> QA & Stats </b></a>"
        )

        self.classes_table = ClassesTable(project_id=project_id)
        if len(classes) > 0:
            self.classes_table.select_classes(classes)
        else:
            self.classes_table.select_all()

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend(
            [qa_stats_text, self.classes_table, self.validator_text, self.button]
        )
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
        return [self.classes_table]

    def get_selected_classes(self) -> list:
        return self.classes_table.get_selected_classes()

    def set_classes(self, classes) -> None:
        self.classes_table.select_classes(classes)

    def select_all_classes(self) -> None:
        self.classes_table.select_all()

    def validate_step(self) -> bool:
        self.validator_text.hide()

        if len(self.classes_table.project_meta.obj_classes) == 0:
            self.validator_text.set(text="Project has no classes", status="error")
            self.validator_text.show()
            return False

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
                    status = "warning"

            class_text = "class" if n_classes == 1 else "classes"
            self.validator_text.set(
                text=f"Selected {n_classes} {class_text}{warning_text}", status=status
            )
        self.validator_text.show()
        return n_classes > 0
