from typing import Any, Dict, List, Union

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
import yaml
from supervisely import Api
from supervisely.app.widgets import (
    Card,
    Container,
    Field,
    RadioTabs,
    SelectString,
    Widget,
)
from supervisely.app.widgets.experiment_selector.experiment_selector import (
    ExperimentSelector,
)
from supervisely.app.widgets.pretrained_models_selector.pretrained_models_selector import (
    PretrainedModelsSelector,
)
from supervisely.nn.experiments import get_experiment_infos
from supervisely.nn.inference.gui.serving_gui import ServingGUI
from supervisely.nn.utils import ModelSource, RuntimeType


class ServingGUITemplate(ServingGUI):
    def __init__(
        self, framework_name: str, models: str = None, app_options: str = None
    ):
        self.api = Api.from_env()
        self.team_id = sly_env.team_id()

        self.framework_name = framework_name

        if models is not None:
            self.models = self._load_models(models)
        else:

            self.models = []
        if app_options is not None:
            self.app_options = self._load_app_options(app_options)
        else:
            self.app_options = {}

        self._template_widgets = None
        self._initialize_layout()
        super().__init__()

    def _initialize_layout(self) -> Widget:
        # Pretrained models
        if self.app_options.get("pretrained_models", True) and self.models is not None:
            self.pretrained_models_table = PretrainedModelsSelector(self.models)
        else:
            self.pretrained_models_table = None

        # Custom models
        if self.app_options.get("custom_models", True):
            experiments = get_experiment_infos(
                self.api, self.team_id, self.framework_name
            )
            self.experiment_selector = ExperimentSelector(self.team_id, experiments)
        else:
            self.experiment_selector = None

        # Tabs
        tabs = []
        if self.pretrained_models_table is not None:
            tabs.append(
                (
                    ModelSource.PRETRAINED,
                    "Publicly available models",
                    self.pretrained_models_table,
                )
            )
        if self.experiment_selector is not None:
            tabs.append(
                (
                    ModelSource.CUSTOM,
                    "Models trained in Supervisely",
                    self.experiment_selector,
                )
            )
        titles, descriptions, content = zip(*tabs)
        self.model_source_tabs = RadioTabs(
            titles=titles,
            descriptions=descriptions,
            contents=content,
        )

        # Runtime
        # @TODO: check runtimes lowercase and set like RuntimeType
        supported_runtimes = self.app_options.get(
            "supported_runtimes", [RuntimeType.PYTORCH]
        )
        if supported_runtimes != [RuntimeType.PYTORCH] and supported_runtimes != []:
            self.runtime_select = SelectString(supported_runtimes)
            runtime_field = Field(
                self.runtime_select, "Runtime", "Select a runtime for inference."
            )
        else:
            self.runtime_select = None
            runtime_field = None

        # Layout
        card_widgets = [self.model_source_tabs]
        if runtime_field is not None:
            card_widgets.append(runtime_field)

        card = Card(
            title="Select Model",
            description="Select the model to deploy and press the 'Serve' button.",
            content=Container(widgets=card_widgets),
        )

        self._template_widgets = [card]
        return self._template_widgets

    @property
    def model_source(self) -> str:
        return self.model_source_tabs.get_active_tab()

    @property
    def model_info(self) -> str:
        if self.model_source == ModelSource.PRETRAINED:
            return self.pretrained_models_table.get_selected_row()
        elif self.model_source == ModelSource.CUSTOM:
            return self.experiment_selector.get_selected_experiment_info()

    @property
    def model_name(self) -> str:
        if self.model_source == ModelSource.PRETRAINED:
            selected_row = self.pretrained_models_table.get_selected_row()
            model_meta = selected_row.get("meta", {})
            model_name = model_meta.get("model_name", None)
        else:
            selected_row = self.experiment_selector.get_selected_experiment_info()
            model_name = selected_row.get("model_name", None)
        return model_name

    @property
    def model_files(self) -> List[str]:
        if self.model_source == ModelSource.PRETRAINED:
            selected_row = self.pretrained_models_table.get_selected_row()
            model_meta = selected_row.get("meta", {})
            model_files = model_meta.get("model_files", [])
        else:
            selected_row = self.experiment_selector.get_selected_experiment_info()
            model_files = selected_row.get("model_files", [])
        return model_files

    @property
    def runtime(self) -> str:
        if self.runtime_select is not None:
            return self.runtime_select.get_value()
        else:
            return RuntimeType.PYTORCH

    def get_ui(self) -> Widget:
        self._template_widgets.extend(
            [
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
            ]
        )
        return Container(widgets=self._template_widgets, gap=3)

    def get_params_from_gui(self) -> Dict[str, Any]:
        return {
            "model_files": self.model_files,
            "model_info": self.model_info,
            "device": self.device,
        }

    # Loaders
    def _load_models(self, models: str) -> List[Dict[str, Any]]:
        """
        Loads models from the provided file or list of model configurations.
        """
        if isinstance(models, str):
            if sly_fs.file_exists(models) and sly_fs.get_file_ext(models) == ".json":
                models = sly_json.load_json_file(models)
            else:
                raise ValueError("File not found or invalid file format.")
        else:
            raise ValueError(
                "Invalid models file. Please provide a valid '.json' file with list of model configurations."
            )

        if not isinstance(models, list):
            raise ValueError("models parameters must be a list of dicts")
        for item in models:
            if not isinstance(item, dict):
                raise ValueError(f"Each item in models must be a dict.")
            model_meta = item.get("meta")
            if model_meta is None:
                raise ValueError(
                    "Model metadata not found. Please update provided models parameter to include key 'meta'."
                )
            model_files = model_meta.get("model_files")
            if model_files is None:
                raise ValueError(
                    "Model files not found in model metadata. "
                    "Please update provided models oarameter to include key 'model_files' in 'meta' key."
                )
        return models

    def _load_app_options(self, app_options: str = None) -> Dict[str, Any]:
        """
        Loads the app_options parameter to ensure it is in the correct format.
        """
        if app_options is None:
            return {}

        if isinstance(app_options, str):
            if sly_fs.file_exists(app_options) and sly_fs.get_file_ext(app_options) in [
                ".yaml",
                ".yml",
            ]:
                with open(app_options, "r") as file:
                    app_options = yaml.safe_load(file)
        else:
            raise ValueError(
                "Invalid app_options file provided. Please provide a valid '.yaml' or '.yml' file with app_options."
            )
        if not isinstance(app_options, dict):
            raise ValueError("app_options must be a dict")
        return app_options
