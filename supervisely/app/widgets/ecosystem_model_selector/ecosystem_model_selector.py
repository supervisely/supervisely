from typing import Dict, List, Tuple

import pandas as pd

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.exceptions import show_dialog
from supervisely.app.widgets import Widget
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.dropdown_checkbox_selector.dropdown_checkbox_selector import (
    DropdownCheckboxSelector,
)
from supervisely.app.widgets.fast_table.fast_table import FastTable


class EcosystemModelSelector(Widget):
    class COLUMN:
        MODEL_NAME = "name"
        FRAMEWORK = "framework"
        TASK_TYPE = "task"
        PARAMETERS = "params (M)"
        # TODO: support metrics for different tasks
        MAP = "mAP"

    COLUMNS = [
        str(COLUMN.MODEL_NAME),
        str(COLUMN.FRAMEWORK),
        str(COLUMN.TASK_TYPE),
        str(COLUMN.PARAMETERS),
        str(COLUMN.MAP),
    ]

    def __init__(self, frameworks: List[str] = None, task_types: List[str] = None, models: List[Dict] = None, api: Api = None, widget_id: str = None):
        if api is None:
            api = Api()
        self.api = api
        self.frameworks = None
        self.task_types = None
        self.models = None

        self.set(frameworks, task_types, models)

        self.framework_filter = DropdownCheckboxSelector(items=[], label="Frameworks")
        self.task_filter = DropdownCheckboxSelector(items=[], label="Task Types")
        self.table = FastTable(columns=self.COLUMNS, page_size=10, is_radio=True, header_right_content=Container(widgets=[self.framework_filter, self.task_filter], direction="horizontal"))
        self.refresh_table()
        self.table.set_filter(self._filter_function)

        @self.framework_filter.value_changed
        def _framework_filter_changed(frameworks: List[DropdownCheckboxSelector.Item]):
            self.frameworks = [f.id for f in frameworks] if frameworks else None
            task_types = [t.id for t in self.task_filter.get_selected()]
            self.task_types = task_types if task_types else None
            self.table.filter((self.frameworks, self.task_types))

        @self.task_filter.value_changed
        def _task_filter_changed(task_types: List[DropdownCheckboxSelector.Item]):
            self.task_types = [t.id for t in task_types] if task_types else None
            frameworks = [f.id for f in self.framework_filter.get_selected()]
            self.frameworks = frameworks if frameworks else None
            self.table.filter((self.frameworks, self.task_types))

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _filter_function(
        self, data: pd.DataFrame, filter_value: Tuple[List[str], List[str]]
    ) -> pd.DataFrame:
        try:
            frameworks, task_types = filter_value
            if not frameworks and not task_types:
                return data
            filtered_models = set()
            for idx, model in enumerate(self.models):
                should_add = True
                if frameworks and model["framework"] not in frameworks:
                    should_add = False
                if task_types and model["task"] not in task_types:
                    should_add = False
                if should_add:
                    filtered_models.add(idx)

            filtered_data = data.iloc[sorted(filtered_models)]
            filtered_data.reset_index(inplace=True, drop=True)
        except Exception as e:
            logger.error(f"Error during filtering: {e}", exc_info=True)
            show_dialog(title="Filtering Error", description=str(e), status="error")
            return data
        return filtered_data

    def _filter_models(self, models: List[Dict], frameworks: List[str], task_types: List[str]) -> List[Dict]:
        if frameworks is None and task_types is None:
            return models

        filtered_models = []
        for model in models:
            if frameworks and model["framework"] not in frameworks:
                continue
            if task_types and model["model"]["task"] not in task_types:
                continue
            filtered_models.append(model)
        return filtered_models

    def set(self, frameworks: List[str] = None, task_types: List[str] = None, models: List[Dict] = None):
        self.frameworks = frameworks
        self.task_types = task_types
        if models is None:
            models = self.api.nn.ecosystem_models_api.list_models()
        self.models = models
        self.models = self._filter_models(self.models, self.frameworks, self.task_types)

    def _map_from_model(self, model: Dict):
        try:
            map = model.get("evaluation", {}).get("metrics", {}).get("mAP", None)
            if map is None:
                return None
            return float(map)
        except:
            return None

    def _params_from_model(self, model: Dict):
        try:
            params = model.get("paramsM", None)
            if params is None:
                return None
            return float(params)
        except:
            return None

    def _get_table_data(self, models: List[Dict] = None) -> List[Dict]:
        if models is None:
            models = self.models
        data = [
            {
                self.COLUMN.FRAMEWORK: model_data["framework"],
                self.COLUMN.MODEL_NAME: model_data["name"],
                self.COLUMN.TASK_TYPE: model_data["task"],
                self.COLUMN.PARAMETERS: self._params_from_model(model_data),
                self.COLUMN.MAP: self._map_from_model(model_data),
            }
            for model_data in models
        ]
        return data

    def refresh_table(self):
        data = self._get_table_data()
        df = pd.DataFrame.from_records(data=data, columns=self.COLUMNS)
        self.table.read_pandas(df)
        unique_frameworks = df[self.COLUMN.FRAMEWORK].unique().tolist()
        self.framework_filter.set(items=[DropdownCheckboxSelector.Item(id=f, name=f) for f in unique_frameworks])
        unique_task_types = df[self.COLUMN.TASK_TYPE].unique().tolist()
        self.task_filter.set(items=[DropdownCheckboxSelector.Item(id=t, name=t) for t in unique_task_types])

    def get_selected(self):
        idx = self.table.get_selected_row().row_index
        models = self._filter_models(self.models, self.frameworks, self.task_types)
        return models[idx]

    def select(self, index: int):
        self.table.select_row(index)

    def select_framework_and_model_name(self, framework: str, model_name: str):
        for idx, model in enumerate(self.models):
            if model["name"] == model_name and model["framework"] == framework:
                self.table.select_row(idx)
                return
        raise ValueError(f"Model with framework `{framework}` and name '{model_name}' not found.")

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def to_html(self):
        return self.table.to_html()

    def disable(self):
        return self.table.disable()

    def enable(self):
        return self.table.enable()

    def hide(self):
        return self.table.hide()

    def show(self):
        return self.table.show()

    @property
    def loading(self):
        return self.table.loading

    @loading.setter
    def loading(self, value: bool):
        self.table.loading = value
