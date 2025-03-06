from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.app.widgets import Button, Card, ClassesTable, Container, Text


class ClassesSelector:
    title = "Classes Selector"
    description = "Select classes that will be used for training"
    lock_message = "Select training and validation splits to unlock"

    def __init__(self, project_id: int, classes: list, app_options: dict = {}):
        # Init widgets
        self.qa_stats_text = None
        self.classes_table = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        # GUI Components
        if is_development() or is_debug_with_sly_net():
            qa_stats_link = abs_url(f"projects/{project_id}/stats/datasets")
        else:
            qa_stats_link = f"/projects/{project_id}/stats/datasets"
        self.qa_stats_text = Text(
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
            [self.qa_stats_text, self.classes_table, self.validator_text, self.button]
        )
        # -------------------------------- #

        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
            collapsable=self.app_options.get("collapsable", False),
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

        project_classes = self.classes_table.project_meta.obj_classes
        if len(project_classes) == 0:
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
            message = "Please select at least one class"
            status = "error"
        else:
            class_text = "class" if n_classes == 1 else "classes"
            message = f"Selected {n_classes} {class_text}"
            status = "success"
            if empty_classes:
                intersections = set(selected_classes).intersection(empty_classes)
                if intersections:
                    class_text = "class" if len(intersections) == 1 else "classes"
                    message += (
                        f". Selected {class_text} have no annotations: {', '.join(intersections)}"
                    )
                    status = "warning"

        self.validator_text.set(text=message, status=status)
        self.validator_text.show()
        return n_classes > 0
