from typing import List, Dict, Optional, Union, Callable
import supervisely.app.widgets as Widgets
from supervisely.app.content import StateJson


class InferenceGUI:
    @property
    def serve_button(self) -> Widgets.Widget:
        # return self.serve_button
        raise NotImplementedError("Have to be implemented in child class")

    def get_device(self) -> str:
        # return "cpu"
        raise NotImplementedError("Have to be implemented in child class")

    def set_deployed(self) -> None:
        raise NotImplementedError("Have to be implemented in child class")

    def get_ui(self) -> Widgets.Widget:
        # return Widgets.Container(widgets_list)
        raise NotImplementedError("Have to be implemented in child class")


class SimpleInferenceGUI(InferenceGUI):
    def __init__(
        self,
        models: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]],
    ):
        if isinstance(models, dict):
            self._support_submodels = True
        else:
            self._support_submodels = False
        self._models = models
        self._support_custom_models = True  # TODO: temporary solution

        device_values = []
        device_names = []
        try:
            import torch

            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
                for i in range(len(gpus)):
                    device_values.append(f"cuda:{i}")
                    device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")
            else:
                device_values = ["cpu"]
                device_names = ["CPU"]
        except:
            device_values = ["cpu"]
            device_names = ["CPU"]

        self._device_select = Widgets.SelectString(values=device_values, labels=device_names)
        self._device_field = Widgets.Field(self._device_select, title="Device")
        if self._support_submodels:
            self._model_select = Widgets.SelectString(list(models.keys()))
            selected_model = self._model_select.get_value()
            cols = list(models[selected_model][0].keys())
            rows = [list(model.values()) for model in models[selected_model]]

            @self._model_select.value_changed
            def update_table(value):
                cols = list(self._models[value][0].keys())
                rows = [list(model.values()) for model in self._models[value]]
                self._models_table.columns = cols
                self._models_table.rows = rows

        else:
            cols = list(models[0].keys())
            rows = [list(model.values()) for model in models]

        self._models_table = Widgets.RadioTable(cols, rows)
        self._serve_button = Widgets.Button("SERVE")
        self._success_label = Widgets.DoneLabel()
        self._success_label.hide()
        if self._support_custom_models:
            self._model_path_input = Widgets.Input(
                placeholder="Path to model file or folder in Team Files"
            )
            self._model_path_field = Widgets.Field(
                self._model_path_input,
                title="Path to model file/folder",
                description="Copy path in Team Files",
            )
            pretrained_tab = []
            if self._support_submodels:
                pretrained_tab.append(self._model_select)
            pretrained_tab.append(self._models_table)
            self._tabs = Widgets.RadioTabs(
                titles=["Pretrained models", "Custom weights"],
                contents=[Widgets.Container(pretrained_tab), self._model_path_field],
            )

    def get_device(self) -> str:
        return self._device_select.get_value()

    def get_model_info(self) -> Union[Dict[str, str], Dict[str, Dict[str, str]]]:
        model_cols = self._models_table.columns
        model_row = self._models_table.get_selected_row()
        row_dict = {col: val for col, val in zip(model_cols, model_row)}
        if self._support_submodels:
            model_group_name = self._model_select.get_value()
            row_dict = {model_group_name: row_dict}
        return row_dict

    def get_model_source(self) -> str:
        if not self._support_custom_models:
            return "Pretrained models"
        return self._tabs.get_active_tab()

    def get_custom_model_link(self) -> str:
        if not self._support_custom_models:
            return None  # TODO: or raise Error?
        return self._model_path_input.get_value()

    @property
    def serve_button(self) -> Widgets.Widget:
        return self._serve_button

    def set_deployed(self):
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_value().upper()} device"
        self._success_label.show()

    def get_ui(self) -> Widgets.Widget:
        widgets = []
        if self._support_custom_models:
            widgets.append(self._tabs)
        else:
            pretrained_tab = []
            if self._support_submodels:
                pretrained_tab.append(self._model_select)
            pretrained_tab.append(self._models_table)
            widgets.append(Widgets.Container(pretrained_tab))
        widgets.extend(
            [
                self._device_field,
                self._serve_button,
                self._success_label,
            ]
        )
        return Widgets.Container(widgets)
