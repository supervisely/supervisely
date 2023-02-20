from typing import List, Dict, Union, Optional
import supervisely.app.widgets as Widgets
from supervisely.task.progress import Progress


class BaseInferenceGUI:
    @property
    def serve_button(self) -> Widgets.Button:
        # return self._serve_button
        raise NotImplementedError("Have to be implemented in child class")

    @property
    def download_progress(self) -> Widgets.Progress:
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
        support_pretrained_models: Optional[bool],
        support_custom_models: Optional[bool],
        add_content_to_pretrained_tab: Optional[Widgets.Widget] = None,
        add_content_to_custom_tab: Optional[Widgets.Widget] = None,
    ):
        if isinstance(models, dict):
            self._support_submodels = True
        else:
            self._support_submodels = False
        self._models = models
        if not support_pretrained_models and not support_custom_models:
            raise ValueError(
                """
                You should provide either pretrained models via get_models() 
                function or allow to use custom models via support_custom_models().
                """
            )
        self._support_custom_models = support_custom_models
        self._support_pretrained_models = support_pretrained_models

        device_values = []
        device_names = []
        try:
            import torch

            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
                for i in range(gpus):
                    device_values.append(f"cuda:{i}")
                    device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")
        except:
            pass
        device_values.append("cpu")
        device_names.append("CPU")

        self._device_select = Widgets.SelectString(values=device_values, labels=device_names)
        self._device_field = Widgets.Field(self._device_select, title="Device")
        self._serve_button = Widgets.Button("SERVE")
        self._success_label = Widgets.DoneLabel()
        self._success_label.hide()
        self._download_progress = Widgets.Progress("Downloading model...", hide_on_finish=True)
        self._download_progress.hide()
        self._change_model_button = Widgets.Button(
            "STOP AND CHOOSE ANOTHER MODEL", button_type="danger"
        )
        self._change_model_button.hide()
        self.custom_model_type = "file"  # ['file' or 'folder']

        tabs_titles = []
        tabs_contents = []
        tabs_descriptions = []

        if self._support_pretrained_models:
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
                    items_right_text=paper_and_year,
                    items_links=model_links,
                    filterable=True,
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
                        model_key
                        for model_key in self._models[selected_model]["checkpoints"][0].keys()
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

            pretrained_tab_content = []
            if self._support_submodels:
                pretrained_tab_content.append(self._model_select)
            pretrained_tab_content.append(self._models_table)
            # add user widget after the table
            widget_to_add = add_content_to_pretrained_tab(self)
            if widget_to_add is not None and not support_pretrained_models:
                raise ValueError(
                    "You can provide content to pretrained models tab only If get_models() is not empty list."
                )
            self.add_to_pretrained_tab = widget_to_add
            if self.add_to_pretrained_tab is not None:
                pretrained_tab_content.append(self.add_to_pretrained_tab)
            tabs_titles.append("Pretrained models")
            tabs_contents.append(Widgets.Container(pretrained_tab_content))
            tabs_descriptions.append("Models trained outside Supervisely")

        if self._support_custom_models:
            self.model_path_input = Widgets.Input(placeholder="Path to model in Team Files")
            custom_tab_widgets = []
            # add user widget to top of tab
            widget_to_add = add_content_to_custom_tab(self)

            if widget_to_add is not None and not support_custom_models:
                raise ValueError(
                    "You can provide content to custom models tab only If support_custom_models() returned True."
                )

            self._model_path_field = Widgets.Field(
                self.model_path_input,
                title=f"Path to model {self.custom_model_type}",
                description="Copy path in Team Files",
            )

            self.add_to_custom_tab = widget_to_add
            if self.add_to_custom_tab is not None:
                custom_tab_widgets.append(self.add_to_custom_tab)
            custom_tab_widgets.append(self._model_path_field)
            custom_tab_content = Widgets.Container(custom_tab_widgets)
            tabs_titles.append("Custom models")
            tabs_contents.append(custom_tab_content)
            tabs_descriptions.append("Models trained in Supervisely and located in Team Files")
            # 1. инитится гуи, в него передается уже виджет который зависит. Значит надо как-то привязать позже.
            # 2.

        self._tabs = Widgets.RadioTabs(
            titles=tabs_titles,
            contents=tabs_contents,
            descriptions=tabs_descriptions,
        )

        @self._change_model_button.click
        def change_model():
            self._success_label.text = ""
            self._success_label.hide()
            self._serve_button.show()
            self._device_select.enable()
            self._change_model_button.hide()
            if self._support_pretrained_models:
                self._models_table.enable()
            if self._support_custom_models:
                self._model_path_input.enable()
            Progress("model deployment canceled", 1).iter_done_report()

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
        model_row = self._models_table.get_selected_row_index()
        if not self._support_submodels:
            checkpoint_info = self._models[model_row]
        else:
            selected_model = self._model_select.get_value()
            checkpoint_info = self._models[selected_model]["checkpoints"][model_row]
        return checkpoint_info

    def get_model_source(self) -> str:
        if not self._support_custom_models:
            return "Pretrained models"
        elif not self._support_pretrained_models:
            return "Custom models"
        return self._tabs.get_active_tab()

    def get_custom_link(self) -> str:
        if not self._support_custom_models:
            return None
        return self._model_path_input.get_value()

    @property
    def serve_button(self) -> Widgets.Button:
        return self._serve_button

    @property
    def download_progress(self) -> Widgets.Progress:
        return self._download_progress

    def set_deployed(self):
        self._success_label.text = f"Model has been successfully loaded on {self._device_select.get_value().upper()} device"
        self._success_label.show()
        self._serve_button.hide()
        self._device_select.disable()
        self._change_model_button.show()
        if self._support_pretrained_models:
            self._models_table.disable()
        if self._support_custom_models:
            self._model_path_input.disable()
        Progress("Model deployed", 1).iter_done_report()

    def get_ui(self) -> Widgets.Widget:
        return Widgets.Container(
            [
                self._tabs,
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
            ]
        )
