"""
GUI module for training application.

This module provides the `TrainGUI` class that handles the graphical user interface (GUI) for managing
training workflows in Supervisely.
"""

import supervisely.io.env as sly_env
from supervisely import Api
from supervisely.app.widgets import Stepper, Widget
from supervisely.nn.training.gui.classes_selector import ClassesSelector
from supervisely.nn.training.gui.hyperparameters_selector import HyperparametersSelector
from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.model_selector import ModelSelector
from supervisely.nn.training.gui.train_val_splits_selector import TrainValSplitsSelector
from supervisely.nn.training.gui.training_process import TrainingProcess
from supervisely.nn.training.gui.utils import set_stepper_step, wrap_button_click
from supervisely.nn.utils import ModelSource


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

        self.framework_name = framework_name
        self.models = models
        self.hyperparameters = hyperparameters
        self.app_options = app_options

        self.team_id = sly_env.team_id()
        self.workspace_id = sly_env.workspace_id()
        self.project_id = sly_env.project_id()  # from app options?
        self.project_info = self._api.project.get_info_by_id(self.project_id)

        # 1. Project selection + Train/val split
        self.input_selector = InputSelector(self.project_info)
        # 2. Select train val splits
        self.train_val_splits_selector = TrainValSplitsSelector(self._api, self.project_id)
        # 3. Select classes
        self.classes_selector = ClassesSelector(self.project_id, [])
        # 4. Model selection
        self.model_selector = ModelSelector(self._api, self.framework_name, self.models)
        # 5. Training parameters (yaml), scheduler preview
        self.hyperparameters_selector = HyperparametersSelector(
            hyperparameters=self.hyperparameters
        )
        # 6. Start Train
        self.training_process = TrainingProcess(app_options)

        # Stepper layout
        self.stepper = Stepper(
            widgets=[
                self.input_selector.card,
                self.train_val_splits_selector.card,
                self.classes_selector.card,
                self.model_selector.card,
                self.hyperparameters_selector.card,
                self.training_process.card,
            ],
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

        # ------------------------------------------------- #

        # Wrappers
        self.training_process_cb = wrap_button_click(
            button=self.hyperparameters_selector.button,
            cards_to_unlock=[],
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
            on_select_click=disable_hyperparams_editor,
            on_reselect_click=disable_hyperparams_editor,
        )

        self.model_selector_cb = wrap_button_click(
            button=self.model_selector.button,
            cards_to_unlock=[self.hyperparameters_selector.card],
            widgets_to_disable=self.model_selector.widgets_to_disable,
            callback=self.hyperparameters_selector_cb,
            validation_text=self.model_selector.validator_text,
            validation_func=self.model_selector.validate_step,
        )

        self.classes_selector_cb = wrap_button_click(
            button=self.classes_selector.button,
            cards_to_unlock=[self.model_selector.card],
            widgets_to_disable=self.classes_selector.widgets_to_disable,
            callback=self.model_selector_cb,
            validation_text=self.classes_selector.validator_text,
            validation_func=self.classes_selector.validate_step,
        )

        self.train_val_splits_selector_cb = wrap_button_click(
            button=self.train_val_splits_selector.button,
            cards_to_unlock=[self.classes_selector.card],
            widgets_to_disable=self.train_val_splits_selector.widgets_to_disable,
            callback=self.classes_selector_cb,
            validation_text=self.train_val_splits_selector.validator_text,
            validation_func=self.train_val_splits_selector.validate_step,
        )

        self.input_selector_cb = wrap_button_click(
            button=self.input_selector.button,
            cards_to_unlock=[self.train_val_splits_selector.card],
            widgets_to_disable=self.input_selector.widgets_to_disable,
            callback=self.train_val_splits_selector_cb,
            validation_text=self.input_selector.validator_text,
            validation_func=self.input_selector.validate_step,
            on_select_click=update_classes_table,
        )
        # ------------------------------------------------- #

        # Handlers

        # Define outside. Used by user in app
        # @self.training_process.start_button.click
        # def start_training():
        #     pass

        # @self.training_process.stop_button.click
        # def stop_training():
        #     pass

        # Other handlers
        @self.hyperparameters_selector.run_model_benchmark_checkbox.value_changed
        def show_mb_speedtest(is_checked: bool):
            self.hyperparameters_selector.toggle_mb_speedtest(is_checked)

        # Buttons
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

        self.layout: Widget = self.stepper

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
    def validate_app_config(self, app_config: dict) -> dict:
        """
        Validate the app configuration dictionary.

        :param app_config: The app configuration dictionary.
        :type app_config: dict
        """
        if not isinstance(app_config, dict):
            raise ValueError("app_config must be a dictionary")

        required_keys = {
            "input": ["project_id"],
            "train_val_splits": ["method"],
            "classes": list,
            "model": ["source"],
            "hyperparameters": (dict, str),  # Allowing dict or str for hyperparameters
        }

        for key, subkeys_or_type in required_keys.items():
            if key not in app_config:
                raise KeyError(f"Missing required key in app_config: {key}")

            if isinstance(subkeys_or_type, list):
                for subkey in subkeys_or_type:
                    if subkey not in app_config[key]:
                        raise KeyError(f"Missing required key in app_config['{key}']: {subkey}")
            elif not isinstance(app_config[key], subkeys_or_type):
                valid_types = (
                    " or ".join([t.__name__ for t in subkeys_or_type])
                    if isinstance(subkeys_or_type, tuple)
                    else subkeys_or_type.__name__
                )
                raise ValueError(f"app_config['{key}'] must be of type {valid_types}")

        model = app_config["model"]
        if model["source"] == "Pretrained models":
            if "model_name" not in model:
                raise KeyError("Missing required key in app_config['model']: model_name")
        elif model["source"] == "Custom models":
            custom_keys = ["task_id", "checkpoint"]
            for key in custom_keys:
                if key not in model:
                    raise KeyError(f"Missing required key in app_config['model']: {key}")

        options = app_config.setdefault(
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
            raise ValueError("app_config['options'] must be a dictionary")

        model_benchmark = options.setdefault(
            "model_benchmark", {"enable": True, "speed_test": True}
        )
        if not isinstance(model_benchmark, dict):
            raise ValueError("app_config['options']['model_benchmark'] must be a dictionary")
        model_benchmark.setdefault("enable", True)
        model_benchmark.setdefault("speed_test", True)

        if not isinstance(options.get("cache_project"), bool):
            raise ValueError("app_config['options']['cache_project'] must be a boolean")

        # Check train val splits
        train_val_splits_settings = app_config.get("train_val_splits")
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
        return app_config

    def load_from_config(self, app_config: dict) -> None:
        """
        Load the GUI state from a configuration dictionary.

        :param app_config: The configuration dictionary.
        :type app_config: dict
        """
        app_config = self.validate_app_config(app_config)

        options = app_config["options"]
        input_settings = app_config["input"]
        train_val_splits_settings = app_config["train_val_splits"]
        classes_settings = app_config["classes"]
        model_settings = app_config["model"]
        hyperparameters_settings = app_config["hyperparameters"]

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
        self.input_selector.set_cache(options["cache_project"])
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
            self.model_selector.custom_models_table.set_by_task_id(model_settings["task_id"])
            active_row = self.model_selector.custom_models_table.get_selected_row()
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

        model_benchmark_settings = options["model_benchmark"]
        self.hyperparameters_selector.set_model_benchmark_checkbox_value(
            model_benchmark_settings["enable"]
        )
        self.hyperparameters_selector.set_speedtest_checkbox_value(
            model_benchmark_settings["speed_test"]
        )
        self.hyperparameters_selector_cb()

    # ----------------------------------------- #