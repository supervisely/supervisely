from typing import List

from supervisely.api.api import Api
from supervisely.nn.utils import ModelSource
from supervisely.nn.task_type import TaskType, AVAILABLE_TASK_TYPES
from supervisely.app.widgets.widget import Widget
from supervisely.app.widgets.input.input import Input
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.select.select import Select

class ExternalModelSelector(Widget):
    """Widget for selecting external models by task type, checkpoint path, config, and classes."""

    def __init__(self, task_types: List[str], need_config: bool = True, need_classes: bool = True, widget_id: str = None):
        """
        :param task_types: List of task types, one of: AVAILABLE_TASK_TYPES.
        :type task_types: List[str]
        :param need_config: If True, show config path input.
        :type need_config: bool
        :param need_classes: If True, show classes file input.
        :type need_classes: bool
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        self.need_config = need_config
        self.need_classes = need_classes
        self.widgets = []

        # Model name input
        self.model_name_input = Input(value="External Model", placeholder="Enter model name")
        self.model_name_field = Field(
            title="Model name",
            description="Model name is used to identify the model in model benchmark and comparison.",
            content=self.model_name_input,
        )
        self.widgets.append(self.model_name_field)

        # Task type selector
        if len(task_types) == 0:
            raise ValueError("At least 1 task type must be provided for external models")
        for task_type in task_types:
            if task_type not in AVAILABLE_TASK_TYPES:
                raise ValueError(f"Invalid task type: {task_type}. Available task types: {AVAILABLE_TASK_TYPES}")

        self.task_types = task_types

        if len(task_types) > 1:
            select_items = [
                Select.Item(value=task_type, label=task_type) for task_type in task_types
            ]
            self.task_type_selector = Select(items=select_items)
            self.task_type_field = Field(
                title="Task type",
                description="Task type of the model",
                content=self.task_type_selector,
            )
            self.widgets.append(self.task_type_field)

        # Checkpoint input
        self.checkpoint_input = Input(placeholder="Enter checkpoint path")
        self.checkpoint_field = Field(title="Checkpoint", description="Path to the checkpoint in Team Files", content=self.checkpoint_input)
        self.widgets.append(self.checkpoint_field)

        # Config input
        if need_config:
            self.config_input = Input(placeholder="Enter config path")
            self.config_field = Field(
                title="Config",
                description="Path to the model config in Team Files",
                content=self.config_input,
            )
            self.widgets.append(self.config_field)

        # Classes input
        if need_classes:
            self.classes_input = Input(placeholder="Enter classes")
            self.classes_field = Field(
                title="Model classes",
                description="Path to '.json' file with classes in Team Files. File should contain classes mapping (e.g {'0': 'person', '1': 'car', '2': 'pizza'})",
                content=self.classes_input,
            )
            self.widgets.append(self.classes_field)

        # Layout
        self.layout = Container(self.widgets)

        super().__init__(widget_id=widget_id)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {
            "modelName": self.get_model_name(),
            "taskType": self.get_task_type(),
            "checkpointPath": self.get_checkpoint_path(),
            "configPath": self.get_config_path(),
            "classesPath": self.get_classes_path(),
        }

    def get_model_name(self):
        return self.model_name_input.get_value()

    def set_model_name(self, model_name: str):
        self.model_name_input.set_value(model_name)

    def get_task_type(self):
        if len(self.task_types) == 1:
            return self.task_types[0]
        return self.task_type_selector.get_value()

    def set_task_type(self, task_type: str):
        self.task_type_selector.set_value(task_type)

    def get_checkpoint_path(self):
        return self.checkpoint_input.get_value()

    def set_checkpoint_path(self, checkpoint_path: str):
        self.checkpoint_input.set_value(checkpoint_path)

    def get_config_path(self):
        if self.need_config:
            return self.config_input.get_value()
        return None

    def set_config_path(self, config_path: str):
        if self.need_config:
            self.config_input.set_value(config_path)
        else:
            raise ValueError("Enable config path is required")

    def get_classes_path(self):
        if self.need_classes:
            return self.classes_input.get_value()
        return None

    def set_classes_path(self, classes_path: str):
        if self.need_classes:
            self.classes_input.set_value(classes_path)
        else:
            raise ValueError("Enable classes path is required")

    def get_deploy_params(self):
        model_info = {"model_name": self.get_model_name(), "task_type": self.get_task_type()}
        model_files = {"checkpoint": self.get_checkpoint_path()}
        if self.need_config:
            model_files["config"] = self.get_config_path()
        if self.need_classes:
            model_files["classes"] = self.get_classes_path()

        deploy_params = {
            "model_info": model_info,
            "model_files": model_files,
            "model_source": ModelSource.EXTERNAL,
        }
        return deploy_params

    def to_html(self):
        return self.layout.to_html()
