from typing import Literal

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
from supervisely import logger
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
from supervisely.nn.artifacts.utils import FrameworkMapper
from supervisely.nn.experiments import get_experiment_infos
from supervisely.nn.utils import ModelSource, _get_model_name
from supervisely._utils import get_filename_from_headers


class ModelSelector:
    title = "Select Model"
    description = "Select a model for training"
    lock_message = "Select previous step to unlock"

    def __init__(self, api: Api, framework: str, models: list, app_options: dict = {}):
        # Init widgets
        self.api = api
        self.pretrained_models_table = None
        self.experiment_selector = None
        self.model_source_tabs = None
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        self.display_widgets = []
        self.app_options = app_options

        model_selector_opts = self.app_options.get("model_selector", {})
        if not isinstance(model_selector_opts, dict):
            model_selector_opts = {}

        self.show_pretrained = True
        self.show_custom = model_selector_opts.get("show_custom", True)

        self.team_id = sly_env.team_id()
        self.models = models

        # GUI Components
        self.pretrained_models_table = PretrainedModelsSelector(self.models)
        experiment_infos = get_experiment_infos(self.api, self.team_id, framework)
        if self.app_options.get("legacy_checkpoints", False):
            try:
                framework_cls = FrameworkMapper.get_framework_cls(framework, self.team_id)
                legacy_experiment_infos = framework_cls.get_list_experiment_info()
                experiment_infos = experiment_infos + legacy_experiment_infos
            except:
                logger.warning(f"Legacy checkpoints are not available for '{framework}'")

        self.experiment_selector = ExperimentSelector(self.api, self.team_id, experiment_infos)

        tab_titles = []
        tab_descriptions = []
        tab_contents = []
        if self.show_pretrained:
            tab_titles.append(ModelSource.PRETRAINED)
            tab_descriptions.append("Publicly available models")
            tab_contents.append(self.pretrained_models_table)
        if self.show_custom:
            tab_titles.append(ModelSource.CUSTOM)
            tab_descriptions.append("Models trained by you in Supervisely")
            tab_contents.append(self.experiment_selector)

        self.model_source_tabs = RadioTabs(
            titles=tab_titles,
            descriptions=tab_descriptions,
            contents=tab_contents,
        )

        if len(tab_titles) > 0:
            self.model_source_tabs.set_active_tab(tab_titles[0])

        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")

        self.display_widgets.extend([self.model_source_tabs, self.validator_text, self.button])
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
            model_name = _get_model_name(selected_row)
        else:
            selected_row = self.experiment_selector.get_selected_experiment_info()
            model_name = selected_row.model_name
        return model_name

    def get_model_info(self) -> dict:
        if self.get_model_source() == ModelSource.PRETRAINED:
            return self.pretrained_models_table.get_selected_row()
        else:
            return self.experiment_selector.get_selected_experiment_info().to_json()

    def get_checkpoint_name(self) -> str:
        if self.get_model_source() == ModelSource.PRETRAINED:
            selected_row = self.pretrained_models_table.get_selected_row()
            meta = selected_row.get("meta", None)
            if meta is not None:
                model_files = meta.get("model_files", None)
                if model_files is not None:
                    checkpoint_name = model_files.get("checkpoint", None)
                    if checkpoint_name is not None:
                        is_url = sly_fs.str_is_url(checkpoint_name)
                        if (not is_url) or checkpoint_name.lower().endswith((".pth", ".pt")):
                            checkpoint_name = sly_fs.get_file_name_with_ext(checkpoint_name)
                        else:
                            checkpoint_name = get_filename_from_headers(checkpoint_name)
                        return checkpoint_name
            return None
        else:
            checkpoint_name = self.experiment_selector.get_selected_checkpoint_name()
        return checkpoint_name

    def get_checkpoint_link(self) -> str:
        if self.get_model_source() == ModelSource.PRETRAINED:
            selected_row = self.pretrained_models_table.get_selected_row()
            meta = selected_row.get("meta", None)
            if meta is not None:
                model_files = meta.get("model_files", None)
                if model_files is not None:
                    checkpoint_link = model_files.get("checkpoint", None)
                    if checkpoint_link is not None:
                        is_url = sly_fs.str_is_url(checkpoint_link)
                        if checkpoint_link.startswith("/experiments/"):
                            is_url = True
                        if is_url:
                            return checkpoint_link
            return None
        else:
            checkpoint_link = self.experiment_selector.get_selected_checkpoint_path()
        return checkpoint_link

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

    def get_selected_task_type(self) -> str:
        if self.get_model_source() == ModelSource.PRETRAINED:
            return self.pretrained_models_table.get_selected_task_type()
        else:
            return self.experiment_selector.get_selected_experiment_info().task_type
