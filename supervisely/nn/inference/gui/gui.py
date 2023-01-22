from typing import List, Dict, Union
import supervisely.app.widgets as Widgets
from supervisely.task.progress import Progress


class BaseInferenceGUI:
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

    def get_location(self) -> Union[str, List[str]]:
        raise NotImplementedError("Have to be implemented in child class")

    def get_custom_files(self, custom_link: str) -> Dict[str, str]:
        # weights_remote_dir = os.path.dirname(custom_link)
        # return {
        #     "weights_file": custom_link,
        #     "config_file": os.path.join(weights_remote_dir, 'config.py')
        # }
        return None


class InferenceGUI(BaseInferenceGUI):
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
        self._support_pretrained_models = True  # TODO: check If get_models() method is implemented
        assert self._support_custom_models or self._support_pretrained_models

        self._weights_file_key = "weights_file"
        self._config_file_key = "config_file"
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
            # TODO: add paper_from and year
            # for model_name, model_data in models.items():
            #     for param_name, param_val in model_data.items():
            #         if param_name == "checkpoints":
            #             continue
            #         elif param_name == "paper_from":

            self._model_select = Widgets.SelectString(list(models.keys()))
            selected_model = self._model_select.get_value()
            cols = [
                model_key
                for model_key in models[selected_model]["checkpoints"][0].keys()
                if model_key not in [self._weights_file_key, self._config_file_key]
            ]
            rows = [
                [
                    value
                    for param_name, value in model.items()
                    if param_name not in [self._config_file_key, self._weights_file_key]
                ]
                for model in models[selected_model]["checkpoints"]
            ]

            @self._model_select.value_changed
            def update_table(selected_model):
                cols = [
                    model_key
                    for model_key in self._models[selected_model]["checkpoints"][0].keys()
                    if model_key not in [self._weights_file_key, self._config_file_key]
                ]
                rows = [
                    [
                        value
                        for param_name, value in model.items()
                        if param_name not in [self._config_file_key, self._weights_file_key]
                    ]
                    for model in self._models[selected_model]["checkpoints"]
                ]
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

    def get_model_info(self) -> Dict[str, Dict[str, str]]:
        if not self._support_submodels:
            return None
        selected_model = self._model_select.get_value()
        selected_model_info = self._models[selected_model].copy()
        del selected_model_info["checkpoints"]
        return {selected_model: selected_model_info}

    def get_checkpoint_info(self) -> Dict[str, str]:
        selected_model = self._model_select.get_value()
        model_row = self._models_table.get_selected_row_index()
        checkpoint_info = self._models[selected_model]["checkpoints"][model_row]
        return checkpoint_info

    def get_model_source(self) -> str:
        if not self._support_custom_models:
            return "Pretrained models"
        return self._tabs.get_active_tab()

    def get_location(self) -> Dict[str, str]:
        location = {}
        if self.get_model_source() == "Pretrained models":
            checkpoint_info = self.get_checkpoint_info()
            if self._weights_file_key in checkpoint_info.keys():
                location[self._weights_file_key] = checkpoint_info[self._weights_file_key]
            if self._config_file_key in checkpoint_info.keys():
                location[self._config_file_key] = checkpoint_info[self._config_file_key]
        elif self.get_model_source() == "Custom models":
            custom_weights_link = self._model_path_input.get_value()
            custom_files = self.get_custom_files(custom_weights_link)
            if custom_files is not None:
                for file_key, file_link in custom_files.items():
                    location[file_key] = file_link
            else:
                location["weights_file"] = custom_weights_link
        return location

    @property
    def serve_button(self) -> Widgets.Widget:
        return self._serve_button

    def set_deployed(self):
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_value().upper()} device"
        self._success_label.show()
        Progress("Model deployed", 1).iter_done_report()

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
