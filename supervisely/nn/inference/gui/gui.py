from typing import List, Dict, Union
import supervisely.app.widgets as Widgets
from supervisely.task.progress import Progress


class BaseInferenceGUI:
    @property
    def serve_button(self) -> Widgets.Button:
        # return self._serve_button
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def download_progress(self) -> Widgets.SlyTqdm:
        # return self._download_progress
        raise NotImplementedError("Have to be implemented in child class")

    def get_device(self) -> str:
        # return "cpu"
        raise NotImplementedError("Have to be implemented in child class")

    def set_deployed(self) -> None:
        raise NotImplementedError("Have to be implemented in child class")

    def get_ui(self) -> Widgets.Widget:
        # return Widgets.Container(widgets_list)
        raise NotImplementedError("Have to be implemented in child class")


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

        device_values = ["cpu"]
        device_names = ["CPU"]
        try:
            import torch

            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
                for i in range(gpus):
                    device_values.append(f"cuda:{i}")
                    device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")
        except:
            pass

        self._device_select = Widgets.SelectString(values=device_values, labels=device_names)
        self._device_field = Widgets.Field(self._device_select, title="Device")
        if self._support_submodels:
            model_papers = []
            model_years = []
            model_links = []
            for _, model_data in models.items():
                for param_name, param_val in model_data.items():
                    if param_name == "paper_from":
                        model_papers.append(param_val)
                    elif param_name == "year":
                        model_years.append(param_val)
                    elif param_name == "config_url":
                        model_links.append(param_val)
            paper_and_year = []
            for paper, year in zip(model_papers, model_years):
                paper_and_year.append(f"{paper} {year}")
            self._model_select = Widgets.SelectString(
                list(models.keys()),
                items_info=paper_and_year,
                items_links=model_links,
            )
            selected_model = self._model_select.get_value()
            cols = list(models[selected_model]["checkpoints"][0].keys())
            rows = [
                [value for param_name, value in model.items()]
                for model in models[selected_model]["checkpoints"]
            ]

            @self._model_select.value_changed
            def update_table(selected_model):
                cols = [
                    model_key for model_key in self._models[selected_model]["checkpoints"][0].keys()
                ]
                rows = [
                    [value for param_name, value in model.items()]
                    for model in self._models[selected_model]["checkpoints"]
                ]

                table_subtitles, cols = self._get_table_subtitles(cols)
                self._models_table.set_data(cols, rows, table_subtitles)

        else:
            cols = list(models[0].keys())
            rows = [list(model.values()) for model in models]

        table_subtitles, cols = self._get_table_subtitles(cols)
        self._models_table = Widgets.RadioTable(cols, rows, subtitles=table_subtitles)
        self._serve_button = Widgets.Button("SERVE")
        self._success_label = Widgets.DoneLabel()
        self._success_label.hide()
        self._download_progress = Widgets.SlyTqdm("Downloading model...", show_percents=True)
        self._download_progress.hide()
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

    def _get_table_subtitles(self, cols):
        # Get subtitles from col's round brackets
        subtitles = {}
        updated_cols = []
        for col in cols:
            guess_brackets = col.split("(")
            if len(guess_brackets) > 1:
                subtitle = guess_brackets[1]
                if ")" not in subtitle:
                    subtitles[col] = None
                    updated_cols.append(col)
                subtitle = subtitle.split(")")[0]
                new_col = guess_brackets[0].strip()
                subtitles[new_col] = subtitle
                updated_cols.append(new_col)
            else:
                subtitles[col] = None
                updated_cols.append(col)
        return subtitles, updated_cols

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

    def get_custom_link(self) -> str:
        if not self._support_custom_models:
            return None
        return self._model_path_input.get_value()

    @property
    def serve_button(self) -> Widgets.Button:
        return self._serve_button

    @property
    def download_progress(self) -> Widgets.SlyTqdm:
        return self._download_progress

    def set_deployed(self):
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_value().upper()} device"
        self._success_label.show()
        self._serve_button.hide()
        self._device_select.disable()
        self._models_table.disable()
        if self._support_custom_models:
            self._model_path_input.disable()
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
                self._download_progress,
                self._success_label,
            ]
        )
        return Widgets.Container(widgets)
