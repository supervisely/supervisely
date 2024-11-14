import supervisely.io.env as sly_env
from supervisely import Api
from supervisely.app.widgets import Stepper, Widget
from supervisely.nn.training.gui.classes_selector import ClassesSelector
from supervisely.nn.training.gui.hyperparameters_selector import HyperparametersSelector
from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.model_selector import ModelSelector
from supervisely.nn.training.gui.training_process import TrainingProcess
from supervisely.nn.training.gui.utils import set_stepper_step, wrap_button_click
from supervisely.nn.utils import ModelSource


class TrainGUI:
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
        self.input_selector = InputSelector(project_info=self.project_info)
        # 2. Select classes
        self.classes_selector = ClassesSelector(project_id=self.project_id, classes=[])
        # 3. Model selection
        self.model_selector = ModelSelector(self._api, self.framework_name, self.models)
        # 4. Training parameters (yaml), scheduler preview
        self.hyperparameters_selector = HyperparametersSelector(
            hyperparameters=self.hyperparameters
        )
        # 5. Start Train
        self.training_process = TrainingProcess()

        # Stepper layout
        self.stepper = Stepper(
            widgets=[
                self.input_selector.card,
                self.classes_selector.card,
                self.model_selector.card,
                self.hyperparameters_selector.card,
                self.training_process.card,
            ],
        )
        # ------------------------------------------------- #

        # Button utils
        def update_classes_table():
            self.classes_selector.set_train_val_datasets(
                train_dataset_id=self.input_selector.get_train_dataset_id(),
                val_dataset_id=self.input_selector.get_val_dataset_id(),
            )

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

        self.input_selector_cb = wrap_button_click(
            button=self.input_selector.button,
            cards_to_unlock=[self.classes_selector.card],
            widgets_to_disable=self.input_selector.widgets_to_disable,
            callback=self.classes_selector_cb,
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
        @self.training_process.logs_button.click
        def show_logs():
            self.training_process.toggle_logs()

        @self.hyperparameters_selector.button.click
        def select_hyperparameters():
            self.hyperparameters_selector_cb()
            set_stepper_step(
                self.stepper,
                self.hyperparameters_selector.button,
                next_pos=5,
            )

        @self.model_selector.button.click
        def select_model():
            self.model_selector_cb()
            set_stepper_step(
                self.stepper,
                self.model_selector.button,
                next_pos=4,
            )

        @self.classes_selector.button.click
        def select_classes():
            self.classes_selector_cb()
            set_stepper_step(
                self.stepper,
                self.classes_selector.button,
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

    # Set GUI from config
    def validate_app_config(self, app_config: dict) -> dict:
        if not isinstance(app_config, dict):
            raise ValueError("app_config must be a dictionary")

        required_keys = {
            "input": ["project_id", "train_dataset_id", "val_dataset_id"],
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

        # Check project and datasets
        project_datasets = self._api.dataset.get_list(app_config["input"]["project_id"])
        project_datasets_ids = [dataset.id for dataset in project_datasets]

        train_ds_id = app_config["input"]["train_dataset_id"]
        val_ds_id = app_config["input"]["val_dataset_id"]

        if train_ds_id not in project_datasets_ids:
            raise ValueError(
                f"Train dataset with given id: '{train_ds_id}' is not found in project"
            )
        if val_ds_id not in project_datasets_ids:
            raise ValueError(f"Val dataset with given id: '{val_ds_id}' is not found in project")

        return app_config

    def load_from_config(self, app_config: dict) -> None:
        app_config = self.validate_app_config(app_config)

        options = app_config["options"]
        input_settings = app_config["input"]
        classes_settings = app_config["classes"]
        model_settings = app_config["model"]
        hyperparameters_settings = app_config["hyperparameters"]

        self._init_input(input_settings, options)
        self._init_classes(classes_settings)
        self._init_model(model_settings)
        self._init_hyperparameters(hyperparameters_settings, options)

    def _init_input(self, input_settings: dict, options: dict) -> None:
        # Set Input
        # Project id will be provided and assigned during _api.app.start
        self.input_selector.set_train_dataset_id(input_settings["train_dataset_id"])
        self.input_selector.set_val_dataset_id(input_settings["val_dataset_id"])
        self.input_selector.set_cache(options["cache_project"])
        self.input_selector_cb()

        # ----------------------------------------- #

    def _init_classes(self, classes_settings: list) -> None:
        # Set Classes
        self.classes_selector.set_classes(classes_settings)
        self.classes_selector_cb()
        # ----------------------------------------- #

    def _init_model(self, model_settings: dict) -> None:
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
            # @TODO: check available task ids prior to setting^^^
            # Can check with get_experiment_infos but not efficient?
            active_row = self.model_selector.custom_models_table.get_selected_row()
            if model_settings["checkpoint"] not in active_row.checkpoints_names:
                raise ValueError(
                    f"Checkpoint '{model_settings['checkpoint']}' not found in selected task"
                )

            active_row.set_selected_checkpoint_by_name(model_settings["checkpoint"])
        self.model_selector_cb()
        # ----------------------------------------- #

    def _init_hyperparameters(self, hyperparameters_settings: dict, options: dict) -> None:
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
