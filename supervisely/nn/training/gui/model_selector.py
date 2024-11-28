from typing import Literal

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    ExperimentSelector,
    PretrainedModelsSelector,
    RadioTabs,
    Text,
)
from supervisely.nn.experiments import get_experiment_infos
from supervisely.nn.utils import ModelSource


class ModelSelector:
    title = "Select Model"
    description = "Select a model for training"
    lock_message = "Select classes to unlock"

    def __init__(self, api: Api, framework: str, models: list, app_options: dict = {}):
        self.team_id = sly_env.team_id()  # get from project id
        self.models = models

        # Pretrained models
        self.pretrained_models_table = PretrainedModelsSelector(self.models)

        experiment_infos = get_experiment_infos(api, self.team_id, framework)
        self.experiment_selector = ExperimentSelector(self.team_id, experiment_infos)
        # Model source tabs
        self.model_source_tabs = RadioTabs(
            titles=[ModelSource.PRETRAINED, ModelSource.CUSTOM],
            descriptions=[
                "Publicly available models",
                "Models trained by you in Supervisely",
            ],
            contents=[self.pretrained_models_table, self.experiment_selector],
        )

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container([self.model_source_tabs, self.validator_text, self.button])
        self.card = Card(
            title=self.title,
            description=self.description,
            content=container,
            lock_message=self.lock_message,
            collapsable=app_options.get("collapsable", False),
        )
        self.card.lock()

    @property
    def widgets_to_disable(self) -> list:
        return [
            self.model_source_tabs,
            self.pretrained_models_table,
            self.experiment_selector,
        ]

    def get_model_source(self) -> str:
        return self.model_source_tabs.get_active_tab()

    def set_model_source(self, source: Literal["Pretrained models", "Custom models"]) -> None:
        self.model_source_tabs.set_active_tab(source)

    def get_model_name(self) -> str:
        if self.get_model_source() == ModelSource.PRETRAINED:
            selected_row = self.pretrained_models_table.get_selected_row()
            model_meta = selected_row.get("meta", {})
            model_name = model_meta.get("model_name", None)
        else:
            selected_row = self.experiment_selector.get_selected_experiment_info()
            model_name = selected_row.get("model_name", None)
        return model_name

    def get_model_info(self) -> dict:
        if self.get_model_source() == ModelSource.PRETRAINED:
            return self.pretrained_models_table.get_selected_row()
        else:
            return self.experiment_selector.get_selected_experiment_info()

    def validate_step(self) -> bool:
        self.validator_text.hide()
        model_info = self.get_model_info()
        if model_info is None or model_info == {}:
            self.validator_text.set(text="Model is not selected", status="error")
            self.validator_text.show()
            return False
        else:
            self.validator_text.set(text="Model is selected", status="success")
            self.validator_text.show()
            return True
