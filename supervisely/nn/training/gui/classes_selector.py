from supervisely._utils import abs_url, is_debug_with_sly_net, is_development
from supervisely.app.widgets import Button, Card, ClassesTable, Container, Text, CheckboxField
from supervisely.nn.training.gui.model_selector import ModelSelector
from typing import List
from supervisely.nn.task_type import TaskType
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.rectangle import Rectangle


class ClassesSelector:
    title = "Classes Selector"
    description = "Select classes that will be used for training"
    lock_message = "Select previous step to unlock"

    def __init__(self, project_id: int, classes: list, model_selector: ModelSelector = None, app_options: dict = {}):
        # Init widgets
        self.model_selector = model_selector
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

        # Auto-convert checkbox
        self.auto_convert_checkbox = CheckboxField(
            title="Auto convert classes to model task type",
            description="If possible, classes with wrong shapes will be converted to the model task type",
            checked=False,
        )

        self.button = Button("Select")
        self.display_widgets.extend(
            [
                self.qa_stats_text,
                self.classes_table,
                self.auto_convert_checkbox,
                self.validator_text,
                self.button,
            ]
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
        return [self.classes_table, self.auto_convert_checkbox]

    def get_selected_classes(self) -> list:
        return self.classes_table.get_selected_classes()

    def set_classes(self, classes) -> None:
        self.classes_table.select_classes(classes)

    def select_all_classes(self) -> None:
        self.classes_table.select_all()

    def get_wrong_shape_classes(self, task_type: str) -> List[str]:
        allowed_geometries = {
            TaskType.OBJECT_DETECTION: {Rectangle},
            TaskType.INSTANCE_SEGMENTATION: {Bitmap},
            TaskType.SEMANTIC_SEGMENTATION: {Bitmap},
            TaskType.POSE_ESTIMATION: {GraphNodes},
        }

        if task_type not in allowed_geometries:
            return []

        selected_classes = self.get_selected_classes()
        wrong_shape_classes = []
        for class_name in selected_classes:
            obj_class = self.classes_table.project_meta.get_obj_class(class_name)
            if obj_class is None:
                continue
            if obj_class.geometry_type not in allowed_geometries[task_type]:
                wrong_shape_classes.append(class_name)

        return wrong_shape_classes

    def validate_step(self) -> bool:
        # @TODO: Handle AnyShape classes
        # @TODO: Handle unconvertable classes (Rectangle -> Bitmap, etc)
        self.validator_text.hide()
        task_type = self.model_selector.get_selected_task_type() if self.model_selector else None

        if len(self.classes_table.project_meta.obj_classes) == 0:
            self.validator_text.set(text="Project has no classes", status="error")
            self.validator_text.show()
            return False

        selected_classes = self.classes_table.get_selected_classes()
        n_classes = len(selected_classes)

        if n_classes == 0:
            self.validator_text.set(text="Please select at least one class", status="error")
            self.validator_text.show()
            return False

        # ----------------------------------------------------------- #
        class_word = "class" if n_classes == 1 else "classes"
        message_parts = [f"Selected {n_classes} {class_word}"]
        status = "success"
        is_valid = True

        empty_classes = [
            row[0]["data"]
            for row in self.classes_table._table_data
            if row[0]["data"] in selected_classes and row[2]["data"] == 0 and row[3]["data"] == 0
        ]
        if empty_classes:
            empty_word = "class" if len(empty_classes) == 1 else "classes"
            message_parts.append(
                f"{empty_word.capitalize()} with no annotations: {', '.join(empty_classes)}"
            )
            status = "warning"

        # ----------------------------------------------------------- #
        wrong_shape_classes = self.get_wrong_shape_classes(task_type)

        if wrong_shape_classes:
            task_specific_texts = {
                TaskType.OBJECT_DETECTION: "Only rectangle shapes are supported for object detection task",
                TaskType.INSTANCE_SEGMENTATION: "Only bitmap shapes are supported for instance segmentation task",
                TaskType.SEMANTIC_SEGMENTATION: "Only bitmap shapes are supported for semantic segmentation task",
                TaskType.POSE_ESTIMATION: "Only keypoint (graph) shape is supported for pose estimation task",
            }

            if self.is_auto_convert_enabled():
                message_parts.append(
                    f"Auto-convert enabled. Incompatible classes will be converted for task '{task_type}'."
                )
                status = "info" if status == "success" else status
                is_valid = True
            else:
                specific_text = task_specific_texts.get(
                    task_type,
                    "Some selected classes have shapes that are incompatible with the chosen model task type.",
                )
                message_parts = [
                    f"Model task type is {task_type}. {specific_text}",
                    "Select compatible classes or enable auto-convert.",
                ]
                status = "error"
                is_valid = False
        else:
            if self.is_auto_convert_enabled():
                message_parts.append("Auto-convert enabled, but no shape conversion required.")

        # ----------------------------------------------------------- #
        self.validator_text.set(text=". ".join(message_parts), status=status)
        self.validator_text.show()
        return is_valid

    def is_auto_convert_enabled(self) -> bool:
        return self.auto_convert_checkbox.is_checked()
