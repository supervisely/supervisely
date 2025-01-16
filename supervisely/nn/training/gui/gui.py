"""
GUI module for training application.

This module provides the `TrainGUI` class that handles the graphical user interface (GUI) for managing
training workflows in Supervisely.
"""

from os import environ

import supervisely.io.env as sly_env
from supervisely import Api, ProjectMeta
from supervisely._utils import is_production
from supervisely.app.widgets import Stepper, Widget
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.nn.task_type import TaskType
from supervisely.nn.training.gui.classes_selector import ClassesSelector
from supervisely.nn.training.gui.hyperparameters_selector import HyperparametersSelector
from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.model_selector import ModelSelector
from supervisely.nn.training.gui.train_val_splits_selector import TrainValSplitsSelector
from supervisely.nn.training.gui.training_artifacts import TrainingArtifacts
from supervisely.nn.training.gui.training_logs import TrainingLogs
from supervisely.nn.training.gui.training_process import TrainingProcess
from supervisely.nn.training.gui.utils import set_stepper_step, wrap_button_click
from supervisely.nn.utils import ModelSource, RuntimeType


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
                self.task_id = "debug-session"

        self.framework_name = framework_name
        self.models = models
        self.hyperparameters = hyperparameters
        self.app_options = app_options
        self.collapsable = app_options.get("collapsable", False)
        self.need_convert_shapes_for_bm = False

        self.team_id = sly_env.team_id(raise_not_found=False)
        self.workspace_id = sly_env.workspace_id(raise_not_found=False)
        self.project_id = sly_env.project_id()
        self.project_info = self._api.project.get_info_by_id(self.project_id)
        self.project_meta = ProjectMeta.from_json(self._api.project.get_meta(self.project_id))

        if self.workspace_id is None:
            self.workspace_id = self.project_info.workspace_id
            environ["WORKSPACE_ID"] = str(self.workspace_id)
        if self.team_id is None:
            self.team_id = self.project_info.team_id
            environ["TEAM_ID"] = str(self.team_id)

        # 1. Project selection + Train/val split
        self.input_selector = InputSelector(self.project_info, self.app_options)
        # 2. Select train val splits
        self.train_val_splits_selector = TrainValSplitsSelector(
            self._api, self.project_id, self.app_options
        )
        # 3. Select classes
        self.classes_selector = ClassesSelector(self.project_id, [], self.app_options)
        # 4. Model selection
        self.model_selector = ModelSelector(
            self._api, self.framework_name, self.models, self.app_options
        )
        # 5. Training parameters (yaml), scheduler preview
        self.hyperparameters_selector = HyperparametersSelector(
            self.hyperparameters, self.app_options
        )
        # 6. Start Train
        self.training_process = TrainingProcess(self.app_options)

        # 7. Training logs
        self.training_logs = TrainingLogs(self.app_options)

        # 8. Training Artifacts
        self.training_artifacts = TrainingArtifacts(self._api, self.app_options)

        # Stepper layout
        self.steps = [
            self.input_selector.card,
            self.train_val_splits_selector.card,
            self.classes_selector.card,
            self.model_selector.card,
            self.hyperparameters_selector.card,
            self.training_process.card,
            self.training_logs.card,
            self.training_artifacts.card,
        ]
        self.stepper = Stepper(
            widgets=self.steps,
        )
        # ------------------------------------------------- #

        # Button utils
        def update_classes_table():
            pass

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
                experiment_name = f"{self.task_id}_{self.project_info.name}_{model_name}"

            if experiment_name == self.training_process.get_experiment_name():
                return
            self.training_process.set_experiment_name(experiment_name)

        def need_convert_class_shapes() -> bool:
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
                    return

                data = self.classes_selector.classes_table._table_data
                selected_classes = set(self.classes_selector.classes_table.get_selected_classes())
                empty = set(r[0]["data"] for r in data if r[2]["data"] == 0 and r[3]["data"] == 0)
                need_convert = set(r[0]["data"] for r in data if _need_convert(r[1]["data"]))

                if need_convert.intersection(selected_classes - empty):
                    self.hyperparameters_selector.model_benchmark_auto_convert_warning.show()
                    self.need_convert_shapes_for_bm = True
                else:
                    self.hyperparameters_selector.model_benchmark_auto_convert_warning.hide()
                    self.need_convert_shapes_for_bm = False

        # ------------------------------------------------- #

        # Wrappers
        self.training_process_cb = wrap_button_click(
            button=self.hyperparameters_selector.button,
            cards_to_unlock=[self.training_logs.card],
            widgets_to_disable=self.training_process.widgets_to_disable,
            callback=None,
            validation_text=self.training_process.validator_text,
            validation_func=self.training_process.validate_step,
        )

        self.hyperparameters_selector_cb = wrap_button_click(
            button=self.hyperparameters_selector.button,
            cards_to_unlock=[self.training_process.card],
            widgets_to_disable=self.hyperparameters_selector.widgets_to_disable,
            callback=self.training_process_cb,
            validation_text=self.hyperparameters_selector.validator_text,
            validation_func=self.hyperparameters_selector.validate_step,
            on_select_click=[disable_hyperparams_editor],
            on_reselect_click=[disable_hyperparams_editor],
            collapse_card=(self.hyperparameters_selector.card, self.collapsable),
        )

        self.model_selector_cb = wrap_button_click(
            button=self.model_selector.button,
            cards_to_unlock=[self.hyperparameters_selector.card],
            widgets_to_disable=self.model_selector.widgets_to_disable,
            callback=self.hyperparameters_selector_cb,
            validation_text=self.model_selector.validator_text,
            validation_func=self.model_selector.validate_step,
            on_select_click=[set_experiment_name, need_convert_class_shapes],
            collapse_card=(self.model_selector.card, self.collapsable),
        )

        self.classes_selector_cb = wrap_button_click(
            button=self.classes_selector.button,
            cards_to_unlock=[self.model_selector.card],
            widgets_to_disable=self.classes_selector.widgets_to_disable,
            callback=self.model_selector_cb,
            validation_text=self.classes_selector.validator_text,
            validation_func=self.classes_selector.validate_step,
            collapse_card=(self.classes_selector.card, self.collapsable),
        )

        self.train_val_splits_selector_cb = wrap_button_click(
            button=self.train_val_splits_selector.button,
            cards_to_unlock=[self.classes_selector.card],
            widgets_to_disable=self.train_val_splits_selector.widgets_to_disable,
            callback=self.classes_selector_cb,
            validation_text=self.train_val_splits_selector.validator_text,
            validation_func=self.train_val_splits_selector.validate_step,
            collapse_card=(self.train_val_splits_selector.card, self.collapsable),
        )

        self.input_selector_cb = wrap_button_click(
            button=self.input_selector.button,
            cards_to_unlock=[self.train_val_splits_selector.card],
            widgets_to_disable=self.input_selector.widgets_to_disable,
            callback=self.train_val_splits_selector_cb,
            validation_text=self.input_selector.validator_text,
            validation_func=self.input_selector.validate_step,
            on_select_click=[update_classes_table],
            collapse_card=(self.input_selector.card, self.collapsable),
        )
        # ------------------------------------------------- #

        # Main Buttons

        # Define outside. Used by user in app
        # @self.training_process.start_button.click
        # def start_training():
        #     pass

        # @self.training_process.stop_button.click
        # def stop_training():
        #     pass

        # ------------------------------------------------- #

        # Select Buttons
        @self.hyperparameters_selector.button.click
        def select_hyperparameters():
            self.hyperparameters_selector_cb()
            set_stepper_step(
                self.stepper,
                self.hyperparameters_selector.button,
                next_pos=6,
            )

        @self.model_selector.button.click
        def select_model():
            self.model_selector_cb()
            set_stepper_step(
                self.stepper,
                self.model_selector.button,
                next_pos=5,
            )

        @self.classes_selector.button.click
        def select_classes():
            self.classes_selector_cb()
            set_stepper_step(
                self.stepper,
                self.classes_selector.button,
                next_pos=4,
            )

        @self.train_val_splits_selector.button.click
        def select_train_val_splits():
            self.train_val_splits_selector_cb()
            set_stepper_step(
                self.stepper,
                self.train_val_splits_selector.button,
                next_pos=3,
            )

        @self.input_selector.button.click
        def select_input():
            self.input_selector_cb()
            set_stepper_step(
                self.stepper,
                self.input_selector.button,
                next_pos=2,
            )

        # ------------------------------------------------- #

        # Other Buttons
        if app_options.get("show_logs_in_gui", False):

            @self.training_logs.logs_button.click
            def show_logs():
                self.training_logs.toggle_logs()

        # Other handlers
        @self.hyperparameters_selector.run_model_benchmark_checkbox.value_changed
        def show_mb_speedtest(is_checked: bool):
            self.hyperparameters_selector.toggle_mb_speedtest(is_checked)
            need_convert_class_shapes()

        # ------------------------------------------------- #

        self.layout: Widget = self.stepper

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
        self.input_selector.button.enable()
        self.train_val_splits_selector.button.enable()
        self.classes_selector.button.enable()
        self.model_selector.button.enable()
        self.hyperparameters_selector.button.enable()

    def disable_select_buttons(self):
        """
        Makes all select buttons in the GUI unavailable for interaction.
        """
        self.input_selector.button.disable()
        self.train_val_splits_selector.button.disable()
        self.classes_selector.button.disable()
        self.model_selector.button.disable()
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

        required_keys = {
            "input": ["project_id"],
            "train_val_split": ["method"],
            "classes": list,
            "model": ["source"],
            "hyperparameters": (dict, str),  # Allowing dict or str for hyperparameters
        }

        for key, subkeys_or_type in required_keys.items():
            if key not in app_state:
                raise KeyError(f"Missing required key in app_state: {key}")

            if isinstance(subkeys_or_type, list):
                for subkey in subkeys_or_type:
                    if subkey not in app_state[key]:
                        raise KeyError(f"Missing required key in app_state['{key}']: {subkey}")
            elif not isinstance(app_state[key], subkeys_or_type):
                valid_types = (
                    " or ".join([t.__name__ for t in subkeys_or_type])
                    if isinstance(subkeys_or_type, tuple)
                    else subkeys_or_type.__name__
                )
                raise ValueError(f"app_state['{key}'] must be of type {valid_types}")

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

    def load_from_app_state(self, app_state: dict) -> None:
        """
        Load the GUI state from app state dictionary.

        :param app_state: The state dictionary.
        :type app_state: dict

        app_state example:

            app_state = {
                "input": {"project_id": 43192},
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
        app_state = self.validate_app_state(app_state)

        options = app_state.get("options", {})
        input_settings = app_state["input"]
        train_val_splits_settings = app_state["train_val_split"]
        classes_settings = app_state["classes"]
        model_settings = app_state["model"]
        hyperparameters_settings = app_state["hyperparameters"]

        self._init_input(input_settings, options)
        self._init_classes(classes_settings)
        self._init_train_val_splits(train_val_splits_settings)
        self._init_model(model_settings)
        self._init_hyperparameters(hyperparameters_settings, options)

    def _init_input(self, input_settings: dict, options: dict) -> None:
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

    def _init_train_val_splits(self, train_val_splits_settings: dict) -> None:
        """
        Initialize the train/val splits selector with the given settings.

        :param train_val_splits_settings: The train/val splits settings.
        :type train_val_splits_settings: dict
        """
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

    def _init_classes(self, classes_settings: list) -> None:
        """
        Initialize the classes selector with the given settings.

        :param classes_settings: The classes settings.
        :type classes_settings: list
        """
        # Set Classes
        self.classes_selector.set_classes(classes_settings)
        self.classes_selector_cb()
        # ----------------------------------------- #

    def _init_model(self, model_settings: dict) -> None:
        """
        Initialize the model selector with the given settings.

        :param model_settings: The model settings.
        :type model_settings: dict
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
