import os
from os.path import join
from typing import Any, Dict, List, Optional, Union

import yaml

import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
from supervisely import Api
from supervisely.app.widgets import (
    Card,
    Container,
    Field,
    RadioTabs,
    SelectString,
    Text,
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
from supervisely.nn.utils import ModelSource, RuntimeType, _get_model_name
from supervisely.nn.experiments import ExperimentInfo


class ServingGUITemplate(ServingGUI):
    def __init__(
        self,
        framework_name: str,
        models: Optional[list] = None,
        app_options: Optional[str] = None,
    ):
        if not isinstance(framework_name, str):
            raise ValueError("'framework_name' must be a string")
        super().__init__()

        self.api = Api.from_env()
        self.team_id = sly_env.team_id()

        self.framework_name = framework_name
        self.models = models
        self.app_options = self._load_app_options(app_options) if app_options else {}

        base_widgets = self._initialize_layout()
        extra_widgets = self._initialize_extra_widgets()

        self.widgets = base_widgets + extra_widgets
        self.card = self._get_card()

    def _get_card(self) -> Card:
        return Card(
            title="Select Model",
            description="Select the model to deploy and press the 'Serve' button.",
            content=Container(widgets=self.widgets, gap=10),
            overflow="unset",
        )

    def _initialize_layout(self) -> List[Widget]:
        # Pretrained models
        use_pretrained_models = self.app_options.get("pretrained_models", True)
        use_custom_models = self.app_options.get("custom_models", True)

        if not use_pretrained_models and not use_custom_models:
            raise ValueError(
                "At least one of 'pretrained_models' or 'custom_models' must be enabled."
            )

        if use_pretrained_models and self.models is not None:
            self.pretrained_models_table = PretrainedModelsSelector(self.models)
        else:
            self.pretrained_models_table = None

        # Custom models
        if use_custom_models:
            experiments = get_experiment_infos(self.api, self.team_id, self.framework_name)
            self.experiment_selector = ExperimentSelector(self.api, self.team_id, experiments)
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
        if tabs:
            titles, descriptions, content = zip(*tabs)
            self.model_source_tabs = RadioTabs(
                titles=titles,
                descriptions=descriptions,
                contents=content,
            )
        else:
            self.model_source_tabs = None

        # Runtime
        default_runtime = RuntimeType.PYTORCH
        available_runtimes = {
            value.lower(): value
            for name, value in vars(RuntimeType).items()
            if not name.startswith("__")  # exclude private attributes
        }
        supported_runtimes_input = self.app_options.get("supported_runtimes", [default_runtime])
        supported_runtimes = [
            available_runtimes[runtime.lower()]
            for runtime in supported_runtimes_input
            if runtime.lower() in available_runtimes
        ]

        if len(supported_runtimes) > 1:
            self.runtime_select = SelectString(supported_runtimes)
            runtime_field = Field(self.runtime_select, "Runtime", "Select a runtime for inference.")
        else:
            self.runtime_select = None
            runtime_field = None

        # Layout
        card_widgets = [self.model_source_tabs]
        if runtime_field is not None:
            card_widgets.append(runtime_field)

        # Runtime exported checkpoint message
        self._export_msg = Text("")
        self._export_msg.hide()
        card_widgets.append(self._export_msg)

        if self.runtime_select is not None:
            self.runtime_select.value_changed(lambda _: self._update_export_message())

        if self.experiment_selector is not None:
            self.experiment_selector.selection_changed(lambda _: self._update_export_message())
            self.experiment_selector.checkpoint_changed(
                lambda row, _: self._update_export_message()
            )

        if self.pretrained_models_table is not None:
            self.pretrained_models_table.model_changed(lambda _: self._update_export_message())

        if self.model_source_tabs is not None:
            self.model_source_tabs.value_changed(lambda _: self._update_export_message())

        self._update_export_message()

        return card_widgets

    def _initialize_extra_widgets(self) -> List[Widget]:
        return []

    @property
    def model_source(self) -> str:
        return self.model_source_tabs.get_active_tab()

    @property
    def model_info(self) -> Dict[str, Any]:
        model_info = self._get_selected_row()
        if isinstance(model_info, ExperimentInfo):
            # model info requires json format
            # to match types of pretrained and custom model info
            model_info = model_info.to_json()
        return model_info

    @property
    def model_name(self) -> Optional[str]:
        if self.model_source == ModelSource.PRETRAINED:
            return _get_model_name(self.model_info)
        else:
            return self.model_info.get("model_name")

    @property
    def model_files(self) -> List[str]:
        if self.model_source == ModelSource.PRETRAINED:
            model_meta = self.model_info.get("meta", {})
            return model_meta.get("model_files", {})
        else:
            experiment_info = self.experiment_selector.get_selected_experiment_info()
            artifacts_dir = experiment_info.artifacts_dir
            model_files = experiment_info.model_files
            full_model_files = {
                name: os.path.join(artifacts_dir, file) for name, file in model_files.items()
            }
            full_model_files["checkpoint"] = self.experiment_selector.get_selected_checkpoint_path()
            return full_model_files

    @property
    def runtime(self) -> str:
        if self.runtime_select is not None:
            return self.runtime_select.get_value()
        return RuntimeType.PYTORCH

    def get_params_from_gui(self) -> Dict[str, Any]:
        return {
            "model_source": self.model_source,
            "model_files": self.model_files,
            "model_info": self.model_info,
            "device": self.device,
            "runtime": self.runtime,
        }

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

    def _get_selected_row(self) -> Dict[str, Any]:
        if self.model_source == ModelSource.PRETRAINED and self.pretrained_models_table:
            return self.pretrained_models_table.get_selected_row()
        elif self.model_source == ModelSource.CUSTOM and self.experiment_selector:
            return self.experiment_selector.get_selected_experiment_info()
        return {}

    def _update_export_message(self):
        self._export_msg.hide()

        runtime = self.runtime
        non_conversion_runtimes = [RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT]
        if runtime not in non_conversion_runtimes:
            return

        if self.model_source == ModelSource.PRETRAINED:
            self._export_msg.set(
                "Checkpoint will be converted before deployment.",
                "info",
            )
            self._export_msg.show()
            return

        checkpoint_name = None
        if self.model_source == ModelSource.CUSTOM and self.experiment_selector is not None:
            selected_experiment_info = self.experiment_selector.get_selected_experiment_info()
            if selected_experiment_info is None:
                return
            checkpoint_name = self.experiment_selector.get_selected_checkpoint_name()
            if checkpoint_name is None:
                return

        model_info = self.model_info or {}
        export_info = model_info.get("export", {})
        available = False
        if isinstance(export_info, dict):
            for key in export_info.keys():
                if key.lower().startswith(runtime.lower()):
                    available = True
                    break
            if checkpoint_name != selected_experiment_info.best_checkpoint:
                available = False

        if available:
            self._export_msg.set(
                "Runtime checkpoint exists – no conversion needed.",
                "info",
            )
        else:
            self._export_msg.set(
                "Checkpoint will be converted before deployment.",
                "info",
            )
        self._export_msg.show()
