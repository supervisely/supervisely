import yaml
from typing import Any, Dict, List

from supervisely.app.widgets import (
    Button,
    Container,
    Card,
    Editor,
    Text,
    Select,
    Field,
    Checkbox,
    Input,
)


class InferenceMode:
    FULL_IMAGE = "Full Image"
    SLIDING_WINDOW = "Sliding Window"


class AddPredictionsMode:
    MERGE_WITH_EXISTING_LABELS = "Merge with existing labels"
    REPLACE_EXISTING_LABELS = "Replace existing labels"
    REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS = "Replace existing labels and save image tags"


class SettingsSelector:
    title = "Settings Selector"
    description = "Select additional settings for model inference"
    lock_message = None

    def __init__(self):
        # Init Step
        self.display_widgets: List[Any] = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.inference_mode_selector = None
        self.inference_mode_field = None
        self.model_prediction_suffix_input = None
        self.model_prediction_suffix_field = None
        self.model_prediction_suffix_checkbox = None
        self.predictions_mode_selector = None
        self.predictions_mode_field = None
        self.inference_settings = None
        # -------------------------------- #

        # Inference Mode
        self.inference_modes = [InferenceMode.FULL_IMAGE, InferenceMode.SLIDING_WINDOW]
        self.inference_mode_selector = Select(
            items=[Select.Item(mode) for mode in self.inference_modes]
        )
        self.inference_mode_selector.set_value(self.inference_modes[0])
        self.inference_mode_field = Field(
            content=self.inference_mode_selector,
            title="Inference mode",
            description="Select how to process images: full images or using sliding window.",
        )
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.inference_mode_field])
        # ----------------------------------- #

        # Class / Tag Suffix
        self.model_prediction_suffix_input = Input(value="model", minlength=1)
        self.model_prediction_suffix_field = Field(
            content=self.model_prediction_suffix_input,
            title="Class and tag suffix",
            description="Suffix that will be added to class and tag names.",
        )
        self.model_prediction_suffix_checkbox = Checkbox("Always add suffix to model predictions")
        # Add widgets to display ------------ #
        self.display_widgets.extend(
            [self.model_prediction_suffix_field, self.model_prediction_suffix_checkbox]
        )
        # ----------------------------------- #

        # Prediction Mode
        self.prediction_modes = [
            AddPredictionsMode.MERGE_WITH_EXISTING_LABELS,
            AddPredictionsMode.REPLACE_EXISTING_LABELS,
            AddPredictionsMode.REPLACE_EXISTING_LABELS_AND_SAVE_IMAGE_TAGS,
        ]
        self.predictions_mode_selector = Select(
            items=[Select.Item(mode) for mode in self.prediction_modes]
        )
        self.predictions_mode_selector.set_value(self.prediction_modes[0])
        self.predictions_mode_field = Field(
            content=self.predictions_mode_selector,
            title="Add predictions mode",
            description="Select how to add predictions to the project: by merging with existing labels or by replacing them.",
        )
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.predictions_mode_field])
        # ----------------------------------- #

        # Inference Settings
        self.inference_settings = Editor("", language_mode="yaml", height_px=300)
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.inference_settings])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.validator_text, self.button])
        # ----------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        self.card.lock()
        # ----------------------------------- #

    @property
    def widgets_to_disable(self) -> list:
        return [
            self.inference_mode_selector,
            self.model_prediction_suffix_input,
            self.model_prediction_suffix_checkbox,
            self.predictions_mode_selector,
            self.inference_settings,
        ]

    def set_inference_settings(self, settings: Dict[str, Any]):
        if isinstance(settings, str):
            self.inference_settings.set_text(settings)
        else:
            self.inference_settings.set_text(yaml.safe_dump(settings))

    def get_inference_settings(self):
        return yaml.safe_load(self.inference_settings.get_text())

    def get_settings(self) -> Dict[str, Any]:
        return {
            "inference_mode": self.inference_mode_selector.get_value(),
            "model_prediction_suffix": self.model_prediction_suffix_input.get_value(),
            "predictions_mode": self.predictions_mode_selector.get_value(),
            "inference_settings": self.inference_settings.get_text(),
        }

    def load_from_json(self, data):
        pass

    def validate_step(self) -> bool:
        return True
