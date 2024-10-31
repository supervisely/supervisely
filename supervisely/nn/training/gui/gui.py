import supervisely.io.env as sly_env
from supervisely import Api
from supervisely.app.widgets import (
    Stepper,
    Widget,
)
from supervisely.nn.training.gui.classes_selector import ClassesSelector
from supervisely.nn.training.gui.input_selector import InputSelector
from supervisely.nn.training.gui.model_selector import ModelSelector
from supervisely.nn.training.gui.hyperparameters_selector import HyperparametersSelector
from supervisely.nn.training.gui.utils import wrap_button_click, set_stepper_step


# training_layout.py
class TrainGUI:
    # 1. Project selection
    #    Train/val split
    # 2. Task type (optional, auto-detect)
    #    Model selection
    # 3. Select classes
    # 4. Training parameters (yaml), scheduler preview
    # 5. Other options
    # 6. Start training button / Stop
    # 7. Progress + charts (tensorboard frame)
    # 8. Upload checkpoints
    # 9. Evaluation report
    def __init__(
        self,
        models: list,
        hyperparameters: str,
        app_options: dict = None,
    ):
        self.api = Api.from_env()

        self.models = models
        self.hyperparameters = hyperparameters
        self.app_options = app_options

        self.team_id = sly_env.team_id()
        self.workspace_id = sly_env.workspace_id()
        self.project_id = sly_env.project_id()  # from app options?
        self.project_info = self.api.project.get_info_by_id(self.project_id)

        # 1. Project selection + Train/val split
        self.input_selector = InputSelector(project_info=self.project_info)
        # 2. Select classes
        self.classes_selector = ClassesSelector(project_id=self.project_id, classes=[])
        # 3. Model selection
        self.model_selector = ModelSelector(models=self.models)
        # 4. Training parameters (yaml), scheduler preview
        self.hyperparameters_selector = HyperparametersSelector(hyperparameters=self.hyperparameters)
        # 5. Start Train
        # ...

        # Stepper layout
        self.stepper = Stepper(
            widgets=[
                self.input_selector.card,
                self.classes_selector.card,
                self.model_selector.card,
                self.hyperparameters_selector.card,
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
            # @TODO: Doesn't work on reselect!
            if  self.hyperparameters_selector.editor.readonly:
                self.hyperparameters_selector.editor.readonly=False
            else:
                self.hyperparameters_selector.editor.readonly=True
        # ------------------------------------------------- #

        # Wrappers
        hyperparameters_selector_cb = wrap_button_click(
            button=self.hyperparameters_selector.button,
            cards_to_unlock=[],
            widgets_to_disable=self.hyperparameters_selector.widgets_to_disable,
            callback=None,
            validation_text=self.hyperparameters_selector.validator_text,
            validation_func=self.hyperparameters_selector.validate_step,
            on_button_click=disable_hyperparams_editor
        )
                
        model_selector_cb = wrap_button_click(
            button=self.model_selector.button,
            cards_to_unlock=[self.hyperparameters_selector.card],
            widgets_to_disable=self.model_selector.widgets_to_disable,
            callback=None,
            validation_text=self.model_selector.validator_text,
            validation_func=self.model_selector.validate_step,
        )

        classes_selector_cb = wrap_button_click(
            button=self.classes_selector.button,
            cards_to_unlock=[self.model_selector.card],
            widgets_to_disable=self.classes_selector.widgets_to_disable,
            callback=model_selector_cb,
            validation_text=self.classes_selector.validator_text,
            validation_func=self.classes_selector.validate_step,
        )

        input_selector_cb = wrap_button_click(
            button=self.input_selector.button,
            cards_to_unlock=[self.classes_selector.card],
            widgets_to_disable=self.input_selector.widgets_to_disable,
            callback=classes_selector_cb,
            validation_text=self.input_selector.validator_text,
            validation_func=self.input_selector.validate_step,
            on_button_click=update_classes_table,
        )
        # ------------------------------------------------- #

        # Handlers
        @self.hyperparameters_selector.button.click
        def select_hyperparameters():
            hyperparameters_selector_cb()
            # set_stepper_step(
            #     self.stepper,
            #     self.hyperparameters_selector.button,
            #     next_pos=5,
            # )
            
        @self.model_selector.button.click
        def select_model():
            model_selector_cb()
            set_stepper_step(
                self.stepper,
                self.model_selector.button,
                next_pos=4,
            )

        @self.classes_selector.button.click
        def select_classes():
            classes_selector_cb()
            set_stepper_step(
                self.stepper,
                self.classes_selector.button,
                next_pos=3,
            )

        @self.input_selector.button.click
        def select_input():
            input_selector_cb()
            set_stepper_step(
                self.stepper,
                self.input_selector.button,
                next_pos=2,
            )


        # ------------------------------------------------- #

        self.layout: Widget = self.stepper
