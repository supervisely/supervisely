from typing import Literal

import supervisely.io.env as sly_env
from supervisely import logger
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    CustomModelsSelector,
    PretrainedModelsSelector,
    RadioTabs,
    Text,
)
from supervisely.nn.artifacts import (
    HRDA,
    RITM,
    RTDETR,
    Detectron2,
    MMClassification,
    MMDetection,
    MMDetection3,
    MMSegmentation,
    UNet,
    YOLOv5,
    YOLOv5v2,
    YOLOv8,
)
from supervisely.nn.artifacts.artifacts import BaseTrainArtifacts
from supervisely.project.download import is_cached


class ModelSelector:
    title = "Model Selector"

    def __init__(self, models: list):
        self.team_id = sly_env.team_id() # get from project id
        self.models = models

        # Pretrained models
        self.pretrained_models_table = PretrainedModelsSelector(self.models)

        # Custom models
        framework = self._detect_framework()
        if framework is not None:
            artifacts: BaseTrainArtifacts = framework(self.team_id)
            custom_artifacts = artifacts.get_list()
        else:
            custom_artifacts = []

        self.custom_models_table = CustomModelsSelector(
            self.team_id,
            custom_artifacts,
            show_custom_checkpoint_path=True,
            custom_checkpoint_task_types=artifacts.get_available_task_types(),
        )
        # Model source tabs
        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=[
                "Publicly available models",
                "Models trained by you in Supervisely",
            ],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        container = Container([self.model_source_tabs, self.validator_text, self.button])
        self.card = Card(
            title="Select Model",
            description="Select a model for training",
            content=container,
            lock_message="Select classes to unlock",
        )
        self.card.lock()

    @property
    def widgets_to_disable(self):
        return [self.model_source_tabs, self.pretrained_models_table, self.custom_models_table]

    def _detect_framework(self):
        app_map = {
            "YOLOv5": YOLOv5,
            "YOLOv5 2.0": YOLOv5v2,
            "YOLOv8": YOLOv8,
            "UNet": UNet,
            "HRDA": HRDA,
            "RITM": RITM,
            "RT-DETR": RTDETR,
            "MMDetection": MMDetection,
            "MMDetection 3.0": MMDetection3,
            "MMSegmentation": MMSegmentation,
            "MMClassification": MMClassification,
            "Detectron2": Detectron2,
        }
        app_name = "Train YOLOv5 2.0"  # sly_env.app_name()
        for app in app_map:
            if app in app_name:
                return app_map[app]
        return None

    def get_model_source(self):
        return self.model_source_tabs.get_active_tab()

    def set_model_source(self, source: Literal["Pretrained models", "Custom models"]):
        self.model_source_tabs.set_active_tab(source)

    def get_model_parameters(self):
        if self.get_model_source() == "Pretrained models":
            return self.pretrained_models_table.get_selected_row()
        else:
            return self.custom_models_table.get_selected_row()

    def validate_step(self):
        self.validator_text.hide()
        model_params = self.get_model_parameters()
        if model_params is None or model_params == {}:
            self.validator_text.set(text="Model is not selected", status="error")
            self.validator_text.show()
            return False
        else:
            self.validator_text.set(text="Model is selected", status="success")
            self.validator_text.show()
            return True
