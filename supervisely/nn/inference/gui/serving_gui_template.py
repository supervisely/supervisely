# app_options:
# pretrained_models: True
# custom_models: True
# runtime_selector: False
from supervisely.nn.inference.gui.serving_gui import ServingGUI


class ServingGUITemplate(ServingGUI):
    def _initialize_layout(self) -> Widget:
        # Pretrained models
        if self.app_options.get("pretrained_models", True) and self.models is not None:
            self.pretrained_models_table = PretrainedModelsSelector(self.models)
        else:
            self.pretrained_models_table = None
        
        # Custom models
        if self.app_options.get("custom_models", True):
            experiments = get_experiment_infos(self.api, self.team_id, self.framework_name)
            self.experiment_selector = ExperimentSelector(self.team_id, experiments)
        else:
            self.experiment_selector = None
        
        # Tabs
        tabs = []
        if self.pretrained_models_table is not None:
            tabs.append((ModelSource.PRETRAINED, "Publicly available models", self.pretrained_models_table))
        if self.experiment_selector is not None:
            tabs.append((ModelSource.CUSTOM, "Models trained in Supervisely", self.experiment_selector))
        titles, descriptions, content = zip(*tabs)
        self.model_source_tabs = RadioTabs(
            titles=titles,
            descriptions=descriptions,
            contents=content,
        )

        # Runtime
        supported_runtimes = self.app_options.get("supported_runtimes", [RuntimeType.PYTORCH])
        if supported_runtimes != [RuntimeType.PYTORCH]:
            self.runtime_select = SelectString(supported_runtimes)
            runtime_field = Field(self.runtime_select, "Runtime", "Select a runtime for inference.")
        else:
            self.runtime_select = None
            runtime_field = None
        
        # Layout
        content = [self.model_source_tabs]
        if runtime_field is not None:
            content.append(runtime_field)
        self.layout = Container(content)

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
    def runtime(self) -> str:
        if self.runtime_select is not None:
            return self.runtime_select.get_value()
        else:
            return RuntimeType.PYTORCH
