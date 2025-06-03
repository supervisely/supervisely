"""
GUI module for training application.

This module provides the `TrainGUI` class that handles the graphical user interface (GUI) for managing
training workflows in Supervisely.
"""

from os import environ
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import supervisely.io.env as sly_env
import supervisely.io.json as sly_json
from supervisely import Api, ProjectMeta
from supervisely._utils import is_production
from supervisely.app.widgets import Button, Card, Stepper, Widget
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.classes_selector import ClassesSelector
from supervisely.nn.training.gui.hyperparameters_selector import HyperparametersSelector
from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.model_selector import ModelSelector
from supervisely.nn.training.gui.tags_selector import TagsSelector
from supervisely.nn.training.gui.train_val_splits_selector import TrainValSplitsSelector
from supervisely.nn.training.gui.training_artifacts import TrainingArtifacts
from supervisely.nn.training.gui.training_logs import TrainingLogs
from supervisely.nn.training.gui.training_process import TrainingProcess
from supervisely.nn.training.gui.utils import set_stepper_step, wrap_button_click
from supervisely.nn.utils import ModelSource, RuntimeType


class StepFlow:
    """
    Manages the flow of steps in the GUI, including wrappers and button handlers.

    Allows flexible configuration of dependencies between steps and automatically
    sets up proper handlers based on layout from app_options.
    """

    def __init__(self, stepper: Stepper, app_options: Dict[str, Any]):
        """
        Initializes the step manager.

        :param stepper: Stepper object for step navigation
        :param app_options: Application options
        """
        self.stepper = stepper
        self.app_options = app_options
        self.collapsable = app_options.get("collapsable", False)
        self.steps = {}  # Step configuration
        self.step_sequence = []  # Step sequence

    def register_step(
        self,
        name: str,
        card: Card,
        button: Optional[Button] = None,
        widgets_to_disable: Optional[List[Widget]] = None,
        validation_text: Optional[Widget] = None,
        validation_func: Optional[Callable] = None,
        position: Optional[int] = None,
    ) -> "StepFlow":
        """
        Registers a step in the GUI.

        :param name: Unique step name
        :param card: Step card widget
        :param button: Button for proceeding to the next step (optional)
        :param widgets_to_disable: Widgets to disable during validation (optional)
        :param validation_text: Widget for displaying validation text (optional)
        :param validation_func: Validation function (optional)
        :param position: Step position in the sequence (starting from 0)
        :return: Current StepFlow object for method chaining
        """
        self.steps[name] = {
            "card": card,
            "button": button,
            "widgets_to_disable": widgets_to_disable or [],
            "validation_text": validation_text,
            "validation_func": validation_func,
            "position": position,
            "next_steps": [],
            "on_select_click": [],
            "on_reselect_click": [],
            "wrapper": None,
            "has_button": button is not None,
        }

        if position is not None:
            while len(self.step_sequence) <= position:
                self.step_sequence.append(None)
            self.step_sequence[position] = name

        return self

    def set_next_steps(self, step_name: str, next_steps: List[str]) -> "StepFlow":
        """
        Sets the list of next steps for the given step.

        :param step_name: Current step name
        :param next_steps: List of names of the next steps
        :return: Current StepFlow object for method chaining
        """
        if step_name in self.steps:
            self.steps[step_name]["next_steps"] = next_steps
        return self

    def add_on_select_actions(
        self, step_name: str, actions: List[Callable], is_reselect: bool = False
    ) -> "StepFlow":
        """
        Adds actions to be executed when a step is selected/reselected.

        :param step_name: Step name
        :param actions: List of functions to execute
        :param is_reselect: True if these are actions for reselection, otherwise False
        :return: Current StepFlow object for method chaining
        """
        if step_name in self.steps:
            key = "on_reselect_click" if is_reselect else "on_select_click"
            self.steps[step_name][key].extend(actions)
        return self

    def build_wrappers(self) -> Dict[str, Callable]:
        """
        Creates wrappers for all steps based on established dependencies.

        :return: Dictionary with created wrappers by step name
        """
        valid_sequence = [s for s in self.step_sequence if s is not None and s in self.steps]

        for step_name in reversed(valid_sequence):
            step = self.steps[step_name]

            cards_to_unlock = []
            for next_step_name in step["next_steps"]:
                if next_step_name in self.steps:
                    cards_to_unlock.append(self.steps[next_step_name]["card"])

            callback = None
            if step["next_steps"] and step["has_button"]:
                for next_step_name in step["next_steps"]:
                    if (
                        next_step_name in self.steps
                        and self.steps[next_step_name].get("wrapper")
                        and self.steps[next_step_name]["has_button"]
                    ):
                        callback = self.steps[next_step_name]["wrapper"]
                        break

            if step["has_button"]:
                wrapper = wrap_button_click(
                    button=step["button"],
                    cards_to_unlock=cards_to_unlock,
                    widgets_to_disable=step["widgets_to_disable"],
                    callback=callback,
                    validation_text=step["validation_text"],
                    validation_func=step["validation_func"],
                    on_select_click=step["on_select_click"],
                    on_reselect_click=step["on_reselect_click"],
                    collapse_card=(step["card"], self.collapsable),
                )

                step["wrapper"] = wrapper

        return {
            name: self.steps[name]["wrapper"]
            for name in self.steps
            if self.steps[name].get("wrapper") and self.steps[name]["has_button"]
        }

    def setup_button_handlers(self) -> None:
        """
        Sets up handlers for buttons of all steps.
        """
        positions = {}
        pos = 1

        for i, step_name in enumerate(self.step_sequence):
            if step_name is not None and step_name in self.steps:
                positions[step_name] = pos
                pos += 1

        for step_name, step in self.steps.items():
            if step_name in positions and step.get("wrapper") and step["has_button"]:

                button = step["button"]
                wrapper = step["wrapper"]
                position = positions[step_name]
                next_position = position + 1

                def create_handler(btn, cb, next_pos):
                    def handler():
                        cb()
                        set_stepper_step(self.stepper, btn, next_pos=next_pos)

                    return handler

                button.click(create_handler(button, wrapper, next_position))

    def build(self) -> Dict[str, Callable]:
        """
        Performs the complete setup of the step system.

        :return: Dictionary with created wrappers by step name
        """
        wrappers = self.build_wrappers()
        self.setup_button_handlers()
        return wrappers


class TrainGUI:
    """
    A class representing the GUI for training workflows.

    This class sets up and manages GUI components such as project selection,
    train/validation split selection, model selection, hyperparameters selection,
    and the training process.

    :param framework_name: Name of the ML framework being used.
    :type framework_name: str
    :param models: List of available models.
    :type models: list
    :param hyperparameters: Hyperparameters for training.
    :type hyperparameters: dict
    :param app_options: Application options for customization.
    :type app_options: dict, optional
    """

    def __init__(
        self,
        framework_name: str,
        models: list,
        hyperparameters: dict,
        app_options: dict = None,
    ):
        self._api = Api.from_env()
        if is_production():
            self.task_id = sly_env.task_id()
        else:
            self.task_id = sly_env.task_id(raise_not_found=False)
            if self.task_id is None:
                self.task_id = -1

        self.framework_name = framework_name
        self.models = models
        self.hyperparameters = hyperparameters
        self.app_options = app_options
        self.collapsable = self.app_options.get("collapsable", False)
        self.need_convert_shapes_for_bm = False

        self.team_id = sly_env.team_id(raise_not_found=False)
        self.workspace_id = sly_env.workspace_id(raise_not_found=False)
        self.project_id = sly_env.project_id()
        self.project_info = self._api.project.get_info_by_id(self.project_id)
        if self.project_info.type is None:
            raise ValueError(f"Project with ID: '{self.project_id}' does not exist or was archived")

        self.project_meta = ProjectMeta.from_json(self._api.project.get_meta(self.project_id))

        if self.workspace_id is None:
            self.workspace_id = self.project_info.workspace_id
            environ["WORKSPACE_ID"] = str(self.workspace_id)
        if self.team_id is None:
            self.team_id = self.project_info.team_id
            environ["TEAM_ID"] = str(self.team_id)

        # ---------- Parse selector options ----------
        self._classes_selector_opts = self.app_options.get("classes_selector", {})
        self._tags_selector_opts = self.app_options.get("tags_selector", {})
        self._train_val_splits_opts = self.app_options.get("train_val_splits_selector", {})
        self._model_selector_opts = self.app_options.get("model_selector", {})

        self.show_classes_selector = self._classes_selector_opts.get("enabled", True)
        self.show_tags_selector = self._tags_selector_opts.get("enabled", False)
        self.show_train_val_splits_selector = self._train_val_splits_opts.get("enabled", True)
        self.show_model_selector = self._model_selector_opts.get("enabled", True)

        # Ensure train_val_splits_methods compatibility
        self._train_val_methods = self._train_val_splits_opts.get("methods", [])
        # --------------------------------------------------------- #

        # ------------------------------------------------- #
        self.steps = []

        # 1. Project selection
        self.input_selector = InputSelector(self.project_info, self.app_options)
        self.steps.append(self.input_selector.card)

        # 2. Train/val split
        self.train_val_splits_selector = None
        if self.show_train_val_splits_selector:
            self.train_val_splits_selector = TrainValSplitsSelector(
                self._api, self.project_id, self.app_options
            )
            self.steps.append(self.train_val_splits_selector.card)

        # 3. Select Classes
        self.classes_selector = None
        if self.show_classes_selector:
            self.classes_selector = ClassesSelector(self.project_id, [], self.app_options)
            self.steps.append(self.classes_selector.card)

        # 4. Select Tags
        self.tags_selector = None
        if self.show_tags_selector:
            self.tags_selector = TagsSelector(self.project_id, [], self.app_options)
            self.steps.append(self.tags_selector.card)

        # 5. Model selection
        self.model_selector = ModelSelector(
            self._api, self.framework_name, self.models, self.app_options
        )
        if self.show_model_selector:
            self.steps.append(self.model_selector.card)

        # 6. Training parameters (yaml)
        self.hyperparameters_selector = HyperparametersSelector(
            self.hyperparameters, self.app_options
        )
        self.steps.append(self.hyperparameters_selector.card)

        # 7. Start Training
        self.training_process = TrainingProcess(self.app_options)
        self.steps.append(self.training_process.card)

        # 8. Training logs
        self.training_logs = TrainingLogs(self.app_options)
        self.steps.append(self.training_logs.card)

        # 9. Training Artifacts
        self.training_artifacts = TrainingArtifacts(self._api, self.app_options)
        self.steps.append(self.training_artifacts.card)

        # Stepper layout
        self.stepper = Stepper(widgets=self.steps)
        # ------------------------------------------------- #

        # Button utils
        def disable_hyperparams_editor():
            if self.hyperparameters_selector.editor.readonly:
                self.hyperparameters_selector.editor.readonly = False
            else:
                self.hyperparameters_selector.editor.readonly = True

        def set_experiment_name():
            model_name = self.model_selector.get_model_name()
            if model_name is None:
                experiment_name = "Enter experiment name"
            else:
                if self.task_id == -1:
                    experiment_name = f"debug_{self.project_info.name}_{model_name}"
                else:
                    experiment_name = f"{self.task_id}_{self.project_info.name}_{model_name}"

            if experiment_name == self.training_process.get_experiment_name():
                return
            self.training_process.set_experiment_name(experiment_name)

        def need_convert_class_shapes() -> bool:
            if self.hyperparameters_selector.run_model_benchmark_checkbox is not None:
                if not self.hyperparameters_selector.run_model_benchmark_checkbox.is_checked():
                    self.hyperparameters_selector.model_benchmark_auto_convert_warning.hide()
                    self.need_convert_shapes_for_bm = False
                else:
                    task_type = self.model_selector.get_selected_task_type()

                    def _need_convert(shape):
                        if task_type == TaskType.OBJECT_DETECTION:
                            return shape != Rectangle.geometry_name()
                        elif task_type in [
                            TaskType.INSTANCE_SEGMENTATION,
                            TaskType.SEMANTIC_SEGMENTATION,
                        ]:
                            return shape == Polygon.geometry_name()
                        return False

                    if self.classes_selector is not None:
                        data = self.classes_selector.classes_table._table_data
                        selected_classes = set(
                            self.classes_selector.classes_table.get_selected_classes()
                        )
                        empty = set(
                            r[0]["data"] for r in data if r[2]["data"] == 0 and r[3]["data"] == 0
                        )
                        need_convert = set(
                            r[0]["data"] for r in data if _need_convert(r[1]["data"])
                        )
                    else:
                        # Set project meta classes when classes selector is disabled
                        selected_classes = set(cls.name for cls in self.project_meta.obj_classes)
                        need_convert = set(
                            obj_class.name
                            for obj_class in self.project_meta.obj_classes
                            if _need_convert(obj_class.geometry_type)
                        )
                        empty = set()

                    if need_convert.intersection(selected_classes - empty):
                        self.hyperparameters_selector.model_benchmark_auto_convert_warning.show()
                        self.need_convert_shapes_for_bm = True
                    else:
                        self.hyperparameters_selector.model_benchmark_auto_convert_warning.hide()
                        self.need_convert_shapes_for_bm = False
            else:
                self.need_convert_shapes_for_bm = False

        def validate_class_shape_for_model_task():
            task_type = self.model_selector.get_selected_task_type()
            if self.classes_selector is not None:
                classes = self.classes_selector.get_selected_classes()
            else:
                classes = list(self.project_meta.obj_classes.keys())

            required_geometries = {
                TaskType.INSTANCE_SEGMENTATION: {Polygon, Bitmap},
                TaskType.SEMANTIC_SEGMENTATION: {Polygon, Bitmap},
                TaskType.POSE_ESTIMATION: {GraphNodes},
            }
            task_specific_texts = {
                TaskType.INSTANCE_SEGMENTATION: "Only polygon and bitmap shapes are supported for segmentation task",
                TaskType.SEMANTIC_SEGMENTATION: "Only polygon and bitmap shapes are supported for segmentation task",
                TaskType.POSE_ESTIMATION: "Only keypoint (graph) shape is supported for pose estimation task",
            }

            if task_type not in required_geometries:
                return

            wrong_shape_classes = [
                class_name
                for class_name in classes
                if self.project_meta.get_obj_class(class_name).geometry_type
                not in required_geometries[task_type]
            ]

            if wrong_shape_classes:
                specific_text = task_specific_texts[task_type]
                message_text = f"Model task type is {task_type}. {specific_text}. Selected classes have wrong shapes for the model task: {', '.join(wrong_shape_classes)}"
                self.model_selector.validator_text.set(
                    text=message_text,
                    status="warning",
                )

        # ------------------------------------------------- #

        self.step_flow = StepFlow(self.stepper, self.app_options)
        position = 0

        # 1. Input selector
        self.step_flow.register_step(
            "input_selector",
            self.input_selector.card,
            self.input_selector.button,
            self.input_selector.widgets_to_disable,
            self.input_selector.validator_text,
            self.input_selector.validate_step,
            position=position,
        )
        position += 1

        # 2. Train/Val splits selector
        if self.show_train_val_splits_selector and self.train_val_splits_selector is not None:
            self.step_flow.register_step(
                "train_val_splits",
                self.train_val_splits_selector.card,
                self.train_val_splits_selector.button,
                self.train_val_splits_selector.widgets_to_disable,
                self.train_val_splits_selector.validator_text,
                self.train_val_splits_selector.validate_step,
                position=position,
            )
            position += 1

        # 3. Classes selector
        if self.show_classes_selector and self.classes_selector is not None:
            self.step_flow.register_step(
                "classes_selector",
                self.classes_selector.card,
                self.classes_selector.button,
                self.classes_selector.widgets_to_disable,
                self.classes_selector.validator_text,
                self.classes_selector.validate_step,
                position=position,
            )
            position += 1

        # 4. Tags selector
        if self.show_tags_selector and self.tags_selector is not None:
            self.step_flow.register_step(
                "tags_selector",
                self.tags_selector.card,
                self.tags_selector.button,
                self.tags_selector.widgets_to_disable,
                self.tags_selector.validator_text,
                self.tags_selector.validate_step,
                position=position,
            )
            position += 1

        # 5. Model selector
        if self.show_model_selector:
            self.step_flow.register_step(
                "model_selector",
                self.model_selector.card,
                self.model_selector.button,
                self.model_selector.widgets_to_disable,
                self.model_selector.validator_text,
                self.model_selector.validate_step,
                position=position,
            ).add_on_select_actions(
                "model_selector",
                [
                    set_experiment_name,
                    need_convert_class_shapes,
                    validate_class_shape_for_model_task,
                ],
            )
            position += 1

        # 6. Hyperparameters selector
        self.step_flow.register_step(
            "hyperparameters_selector",
            self.hyperparameters_selector.card,
            self.hyperparameters_selector.button,
            self.hyperparameters_selector.widgets_to_disable,
            self.hyperparameters_selector.validator_text,
            self.hyperparameters_selector.validate_step,
            position=position,
        ).add_on_select_actions("hyperparameters_selector", [disable_hyperparams_editor])
        self.step_flow.add_on_select_actions(
            "hyperparameters_selector", [disable_hyperparams_editor], is_reselect=True
        )
        position += 1

        # 7. Training process
        self.step_flow.register_step(
            "training_process",
            self.training_process.card,
            None,
            self.training_process.widgets_to_disable,
            self.training_process.validator_text,
            self.training_process.validate_step,
            position=position,
        )
        position += 1

        # 8. Training logs
        self.step_flow.register_step(
            "training_logs",
            self.training_logs.card,
            None,
            self.training_logs.widgets_to_disable,
            self.training_logs.validator_text,
            self.training_logs.validate_step,
            position=position,
        )
        position += 1

        # 9. Training artifacts
        self.step_flow.register_step(
            "training_artifacts",
            self.training_artifacts.card,
            None,
            self.training_artifacts.widgets_to_disable,
            self.training_artifacts.validator_text,
            self.training_artifacts.validate_step,
            position=position,
        )

        # Set dependencies between steps
        has_train_val_splits = (
            self.show_train_val_splits_selector and self.train_val_splits_selector is not None
        )
        has_classes_selector = self.show_classes_selector and self.classes_selector is not None
        has_tags_selector = self.show_tags_selector and self.tags_selector is not None

        # Set step dependency chain
        # 1. Input selector
        prev_step = "input_selector"
        if has_train_val_splits:
            self.step_flow.set_next_steps(prev_step, ["train_val_splits"])
            prev_step = "train_val_splits"
        if has_classes_selector:
            self.step_flow.set_next_steps(prev_step, ["classes_selector"])
            prev_step = "classes_selector"
        if has_tags_selector:
            self.step_flow.set_next_steps(prev_step, ["tags_selector"])
            prev_step = "tags_selector"

        if self.show_model_selector and self.model_selector is not None:
            self.step_flow.set_next_steps(prev_step, ["model_selector"])
            # Model selector -> hyperparameters
            self.step_flow.set_next_steps("model_selector", ["hyperparameters_selector"])
            prev_step = "model_selector"
        else:
            self.step_flow.set_next_steps(prev_step, ["hyperparameters_selector"])

        # 6. Hyperparameters selector -> 7. Training process
        self.step_flow.set_next_steps("hyperparameters_selector", ["training_process"])

        # 7. Training process -> 8. Training logs
        self.step_flow.set_next_steps("training_process", ["training_logs"])

        # 8. Training logs -> 9. Training artifacts
        self.step_flow.set_next_steps("training_logs", ["training_artifacts"])
        # ------------------------------------------------- #

        # Create all wrappers and set button handlers
        wrappers = self.step_flow.build()

        self.input_selector_cb = wrappers.get("input_selector")
        self.train_val_splits_selector_cb = wrappers.get("train_val_splits")
        self.classes_selector_cb = wrappers.get("classes_selector")
        self.tags_selector_cb = wrappers.get("tags_selector")
        self.model_selector_cb = wrappers.get("model_selector")
        self.hyperparameters_selector_cb = wrappers.get("hyperparameters_selector")
        self.training_process_cb = wrappers.get("training_process")
        self.training_logs_cb = wrappers.get("training_logs")
        self.training_artifacts_cb = wrappers.get("training_artifacts")
        # ------------------------------------------------- #

        # Other handlers
        if self.app_options.get("show_logs_in_gui", False):

            @self.training_logs.logs_button.click
            def show_logs():
                self.training_logs.toggle_logs()

        if self.hyperparameters_selector.run_model_benchmark_checkbox is not None:

            @self.hyperparameters_selector.run_model_benchmark_checkbox.value_changed
            def show_mb_speedtest(is_checked: bool):
                self.hyperparameters_selector.toggle_mb_speedtest(is_checked)
                need_convert_class_shapes()

        # ------------------------------------------------- #

        self.layout: Widget = self.stepper

        # (дублирующийся блок был перемещён выше и здесь удалён)

    def set_next_step(self):
        current_step = self.stepper.get_active_step()
        self.stepper.set_active_step(current_step + 1)

    def set_previous_step(self):
        current_step = self.stepper.get_active_step()
        self.stepper.set_active_step(current_step - 1)

    def set_first_step(self):
        self.stepper.set_active_step(1)

    def set_last_step(self):
        self.stepper.set_active_step(len(self.steps))

    def enable_select_buttons(self):
        """
        Makes all select buttons in the GUI available for interaction.
        """
        if self.input_selector is not None:
            self.input_selector.button.enable()
        if self.train_val_splits_selector is not None:
            self.train_val_splits_selector.button.enable()
        if self.classes_selector is not None:
            self.classes_selector.button.enable()
        if self.tags_selector is not None:
            self.tags_selector.button.enable()
        if self.model_selector is not None:
            self.model_selector.button.enable()
        if self.hyperparameters_selector is not None:
            self.hyperparameters_selector.button.enable()

    def disable_select_buttons(self):
        """
        Makes all select buttons in the GUI unavailable for interaction.
        """
        if self.input_selector is not None:
            self.input_selector.button.disable()
        if self.train_val_splits_selector is not None:
            self.train_val_splits_selector.button.disable()
        if self.classes_selector is not None:
            self.classes_selector.button.disable()
        if self.tags_selector is not None:
            self.tags_selector.button.disable()
        if self.model_selector is not None:
            self.model_selector.button.disable()
        if self.hyperparameters_selector is not None:
            self.hyperparameters_selector.button.disable()

    # Set GUI from config
    def validate_app_state(self, app_state: dict) -> dict:
        """
        Validate the app state dictionary.

        :param app_state: The app state dictionary.
        :type app_state: dict
        """
        if not isinstance(app_state, dict):
            raise ValueError("app_state must be a dictionary")

        show_train_val = self.show_train_val_splits_selector
        show_classes = self.show_classes_selector
        show_tags = self.show_tags_selector

        # Basic required keys always needed
        base_required = {
            "model": ["source"],
            "hyperparameters": (dict, str),
        }
        if show_train_val:
            base_required["train_val_split"] = ["method"]
        if show_classes:
            base_required["classes"] = list
        if show_tags:
            base_required["tags"] = list

        for key, subkeys_or_type in base_required.items():
            if key not in app_state:
                raise KeyError(f"Missing required key in app_state: {key}")
            if isinstance(subkeys_or_type, list):
                for subkey in subkeys_or_type:
                    if subkey not in app_state[key]:
                        raise KeyError(f"Missing required key in app_state['{key}']: {subkey}")
            elif not isinstance(app_state[key], subkeys_or_type):
                valid_types = ""
                if isinstance(subkeys_or_type, tuple):
                    type_names = []
                    for t in subkeys_or_type:
                        if hasattr(t, "__name__"):
                            type_names.append(t.__name__)
                        else:
                            type_names.append(type(t).__name__)
                    valid_types = " or ".join(type_names)
                else:
                    if hasattr(subkeys_or_type, "__name__"):
                        valid_types = subkeys_or_type.__name__
                    else:
                        valid_types = type(subkeys_or_type).__name__
                raise ValueError(f"app_state['{key}'] must be of type {valid_types}")

        # Provide defaults for optional sections when selectors are disabled
        if not show_train_val:
            app_state.setdefault("train_val_split", {"method": "random"})
        if not show_classes:
            app_state.setdefault("classes", [])
        if not show_tags:
            app_state.setdefault("tags", [])

        model = app_state["model"]
        if model["source"] == "Pretrained models":
            if "model_name" not in model:
                raise KeyError("Missing required key in app_state['model']: model_name")
        elif model["source"] == "Custom models":
            custom_keys = ["task_id", "checkpoint"]
            for key in custom_keys:
                if key not in model:
                    raise KeyError(f"Missing required key in app_state['model']: {key}")

        options = app_state.setdefault(
            "options",
            {
                "model_benchmark": {
                    "enable": True,
                    "speed_test": True,
                },
                "cache_project": True,
            },
        )

        if not isinstance(options, dict):
            raise ValueError("app_state['options'] must be a dictionary")

        model_benchmark = options.setdefault(
            "model_benchmark", {"enable": True, "speed_test": True}
        )
        if not isinstance(model_benchmark, dict):
            raise ValueError("app_state['options']['model_benchmark'] must be a dictionary")
        model_benchmark.setdefault("enable", True)
        model_benchmark.setdefault("speed_test", True)

        if not isinstance(options.get("cache_project"), bool):
            raise ValueError("app_state['options']['cache_project'] must be a boolean")

        # Check train val splits
        train_val_splits_settings = app_state.get("train_val_split")
        if train_val_splits_settings.get("method") == "datasets":
            dataset_ids = []
            for parents, dataset in self._api.dataset.tree(self.project_id):
                dataset_ids.append(dataset.id)

            train_datasets = train_val_splits_settings.get("train_datasets", [])
            val_datasets = train_val_splits_settings.get("val_datasets", [])

            missing_datasets_ids = []
            for ds_id in train_datasets + val_datasets:
                if ds_id not in dataset_ids:
                    missing_datasets_ids.append(ds_id)

            if len(missing_datasets_ids) > 0:
                missing_datasets_text = ", ".join([str(ds_id) for ds_id in missing_datasets_ids])
                raise ValueError(
                    f"Datasets with ids: {missing_datasets_text} not found in the project"
                )
        elif train_val_splits_settings.get("method") == "tags":
            train_tag = train_val_splits_settings.get("train_tag")
            val_tag = train_val_splits_settings.get("val_tag")
            if not train_tag or not val_tag:
                raise ValueError("train_tag and val_tag must be specified in tags split method")
        elif train_val_splits_settings.get("method") == "random":
            split = train_val_splits_settings.get("split")
            percent = train_val_splits_settings.get("percent")
            if split not in ["train", "val"]:
                raise ValueError("split must be 'train' or 'val'")
            if not isinstance(percent, int) or not 0 < percent < 100:
                raise ValueError("percent must be an integer in range 1 to 99")
        return app_state

    def load_from_app_state(self, app_state: Union[str, dict]) -> None:
        """
        Load the GUI state from app state dictionary.

        :param app_state: The state dictionary.
        :type app_state: dict

        app_state example:

            app_state = {
                "train_val_split": {
                    "method": "random",
                    "split": "train",
                    "percent": 90
                },
                "classes": ["apple"],
                "model": {
                    "source": "Pretrained models",
                    "model_name": "rtdetr_r50vd_coco_objects365"
                },
                "hyperparameters": hyperparameters, # yaml string
                "options": {
                    "model_benchmark": {
                        "enable": True,
                        "speed_test": True
                    },
                    "cache_project": True,
                "export": {
                    "enable": True,
                    "ONNXRuntime": True,
                    "TensorRT": True
                    },
                }
            }
        """
        if isinstance(app_state, str):
            app_state = sly_json.load_json_file(app_state)
        app_state = self.validate_app_state(app_state)

        options = app_state.get("options", {})
        input_settings = app_state.get("input")
        train_val_splits_settings = app_state.get("train_val_split", {})
        classes_settings = app_state.get("classes", [])
        tags_settings = app_state.get("tags", [])
        model_settings = app_state["model"]
        hyperparameters_settings = app_state["hyperparameters"]

        self._init_input(input_settings, options)
        self._init_train_val_splits(train_val_splits_settings, options)
        self._init_classes(classes_settings, options)
        self._init_tags(tags_settings, options)
        self._init_model(model_settings, options)
        self._init_hyperparameters(hyperparameters_settings, options)

    def _init_input(self, input_settings: Union[dict, None], options: dict) -> None:
        """
        Initialize the input selector with the given settings.

        :param input_settings: The input settings.
        :type input_settings: dict
        :param options: The application options.
        :type options: dict
        """
        # Set Input
        self.input_selector.set_cache(options.get("cache_project", True))
        self.input_selector_cb()
        # ----------------------------------------- #

    def _init_train_val_splits(self, train_val_splits_settings: dict, options: dict) -> None:
        """
        Initialize the train/val splits selector with the given settings.

        :param train_val_splits_settings: The train/val splits settings.
        :type train_val_splits_settings: dict
        :param options: The application options.
        :type options: dict
        """
        if self.train_val_splits_selector is None:
            return  # Selector disabled by app options

        if train_val_splits_settings == {}:
            available_methods = self.app_options.get("train_val_splits_methods", [])
            if available_methods == []:
                method = "random"
                train_val_splits_settings = {"method": method, "split": "train", "percent": 80}
            else:
                method = available_methods[0]
                if method == "random":
                    train_val_splits_settings = {"method": method, "split": "train", "percent": 80}
                elif method == "tags":
                    train_val_splits_settings = {
                        "method": method,
                        "train_tag": "train",
                        "val_tag": "val",
                        "untagged_action": "ignore",
                    }
                elif method == "datasets":
                    train_val_splits_settings = {
                        "method": method,
                        "train_datasets": [],
                        "val_datasets": [],
                    }

        split_method = train_val_splits_settings["method"]
        if split_method == "random":
            split = train_val_splits_settings["split"]
            percent = train_val_splits_settings["percent"]
            self.train_val_splits_selector.train_val_splits.set_random_splits(split, percent)
        elif split_method == "tags":
            train_tag = train_val_splits_settings["train_tag"]
            val_tag = train_val_splits_settings["val_tag"]
            untagged_action = train_val_splits_settings["untagged_action"]
            self.train_val_splits_selector.train_val_splits.set_tags_splits(
                train_tag, val_tag, untagged_action
            )
        elif split_method == "datasets":
            train_datasets = train_val_splits_settings["train_datasets"]
            val_datasets = train_val_splits_settings["val_datasets"]
            self.train_val_splits_selector.train_val_splits.set_datasets_splits(
                train_datasets, val_datasets
            )
        self.train_val_splits_selector_cb()

    def _init_classes(self, classes_settings: list, options: dict) -> None:
        """
        Initialize the classes selector with the given settings.

        :param classes_settings: The classes settings.
        :type classes_settings: list
        :param options: The application options.
        :type options: dict
        """
        if self.classes_selector is None:
            return  # Selector disabled by app options

        # Set Classes
        self.classes_selector.set_classes(classes_settings)
        self.classes_selector_cb()
        # ----------------------------------------- #

    def _init_tags(self, tags_settings: list, options: dict) -> None:
        """
        Initialize the tags selector with the given settings.

        :param tags_settings: The tags settings.
        :type tags_settings: list
        :param options: The application options.
        :type options: dict
        """
        if self.tags_selector is None:
            return  # Selector disabled by app options

        # Set Tags
        self.tags_selector.set_tags(tags_settings)
        self.tags_selector_cb()
        # ----------------------------------------- #

    def _init_model(self, model_settings: dict, options: dict) -> None:
        """
        Initialize the model selector with the given settings.

        :param model_settings: The model settings.
        :type model_settings: dict
        :param options: The application options.
        :type options: dict
        """

        # Pretrained
        if model_settings["source"] == ModelSource.PRETRAINED:
            self.model_selector.model_source_tabs.set_active_tab(ModelSource.PRETRAINED)
            self.model_selector.pretrained_models_table.set_by_model_name(
                model_settings["model_name"]
            )

        # Custom
        elif model_settings["source"] == ModelSource.CUSTOM:
            self.model_selector.model_source_tabs.set_active_tab(ModelSource.CUSTOM)
            self.model_selector.experiment_selector.set_by_task_id(model_settings["task_id"])
            active_row = self.model_selector.experiment_selector.get_selected_row()
            if model_settings["checkpoint"] not in active_row.checkpoints_names:
                raise ValueError(
                    f"Checkpoint '{model_settings['checkpoint']}' not found in selected task"
                )

            active_row.set_selected_checkpoint_by_name(model_settings["checkpoint"])
        self.model_selector_cb()
        # ----------------------------------------- #

    def _init_hyperparameters(self, hyperparameters_settings: dict, options: dict) -> None:
        """
        Initialize the hyperparameters selector with the given settings.

        :param hyperparameters_settings: The hyperparameters settings.
        :type hyperparameters_settings: dict
        :param options: The application options.
        :type options: dict
        """
        self.hyperparameters_selector.set_hyperparameters(hyperparameters_settings)

        model_benchmark_settings = options.get("model_benchmark", None)
        if model_benchmark_settings is not None:
            self.hyperparameters_selector.set_model_benchmark_checkbox_value(
                model_benchmark_settings["enable"]
            )
            self.hyperparameters_selector.set_speedtest_checkbox_value(
                model_benchmark_settings["speed_test"]
                if model_benchmark_settings["enable"]
                else False
            )
        export_weights_settings = options.get("export", None)
        if export_weights_settings is not None:
            self.hyperparameters_selector.set_export_onnx_checkbox_value(
                export_weights_settings.get(RuntimeType.ONNXRUNTIME, False)
            )
            self.hyperparameters_selector.set_export_tensorrt_checkbox_value(
                export_weights_settings.get(RuntimeType.TENSORRT, False)
            )
        self.hyperparameters_selector_cb()

    # ----------------------------------------- #
