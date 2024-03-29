from typing import Dict, List, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Widget,
)

# arch_type_key = "archType"


class PretrainedModelsSelector(Widget):
    class Routes:
        ARCH_TYPE_CHANGED = "arch_type_changed"
        TASK_TYPE_CHANGED = "task_type_changed"
        MODEL_CHANGED = "model_changed"

    def __init__(
        self,
        models_list: List[Dict],
        widget_id: str = None,
    ):
        self._api = Api.from_env()

        self._models = models_list
        filtered_models = self._filter_and_sort_models(self._models)

        self._table_data = filtered_models
        self._model_architectures = list(filtered_models.keys())

        self._arch_changes_handled = False
        self._task_changes_handled = False
        self._model_changes_handled = False

        self.__default_selected_arch_type = (
            self._model_architectures[0] if len(self._model_architectures) > 0 else None
        )
        self.__default_selected_task_type = (
            list(filtered_models[self.__default_selected_arch_type].keys())[0]
            if self.__default_selected_arch_type is not None
            else None
        )

        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def models(self) -> List[str]:
        return self._models

    def get_json_data(self) -> Dict:
        return {
            "tableData": self._table_data,
        }

    def get_json_state(self) -> Dict:
        return {
            "selectedRow": 0,
            "selectedTaskType": self.__default_selected_task_type,
            "selectedArchType": self.__default_selected_arch_type,
        }

    def get_selected_task_type(self) -> str:
        return StateJson()[self.widget_id]["selectedTaskType"]

    def get_selected_arch_type(self) -> str:
        return StateJson()[self.widget_id]["selectedArchType"]

    def get_selected_row(self, state=StateJson()) -> Union[List, None]:
        arch_type = self.get_selected_arch_type()
        task_type = self.get_selected_task_type()
        if arch_type is None or task_type is None:
            return

        models = self._table_data[arch_type][task_type]
        if len(models) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = int(widget_actual_state["selectedRow"])
            return models[selected_row_index]

    def get_selected_model_params(self, model_name_column: str = "Model") -> Union[Dict, None]:
        selected_model = self.get_selected_row()
        model_name = selected_model.get(model_name_column)
        if model_name is None:
            raise ValueError(
                "Could not find model name. Make sure you have column 'Model' in your models list."
            )
        checkpoint_filename = f"{model_name.lower()}.pt"
        checkpoint_url = selected_model.get("meta", {}).get("weightsURL")
        if checkpoint_url is None:
            pass

        task_type = self.get_selected_task_type()
        model_params = {
            "model_source": "Pretrained models",
            "task_type": task_type,
            "checkpoint_name": checkpoint_filename,
            "checkpoint_url": checkpoint_url,
        }

        if len(self._model_architectures) > 1:
            arch_type = self.get_selected_arch_type()
            model_params["arch_type"] = arch_type
        return model_params

    def get_selected_row_index(self, state=StateJson()) -> Union[int, None]:
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

    def set_active_arch_type(self, arch_type: str):
        if arch_type not in self._model_architectures:
            raise ValueError(f'Architecture type "{arch_type}" does not exist')
        StateJson()[self.widget_id]["selectedArchType"] = arch_type
        StateJson().send_changes()

    def set_active_task_type(self, task_type: str):
        if task_type not in self._table_data[self.get_selected_arch_type()]:
            raise ValueError(f'Task type "{task_type}" does not exist')
        StateJson()[self.widget_id]["selectedTaskType"] = task_type
        StateJson().send_changes()

    def set_active_row(self, row_index: int):
        if row_index < 0:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()

    def _filter_and_sort_models(self, models: List[Dict]) -> Dict:
        filtered_models = {}

        for model in models:
            # Extract architecture type and task type, defaulting to 'other' if not specified
            arch_type = model.get("meta", {}).get("archType", "other")
            task_type = model.get("meta", {}).get("taskType", model.get("taskType", "other"))

            # Initialize nested dictionary structure if not already present
            if arch_type not in filtered_models:
                filtered_models[arch_type] = {}
            if task_type not in filtered_models[arch_type]:
                filtered_models[arch_type][task_type] = []

            # Add model to the appropriate category
            filtered_models[arch_type][task_type].append(model)

        # Sort the dictionary by architecture and then by task types
        sorted_filtered_models = {
            arch: {task: models for task, models in sorted(tasks.items())}
            for arch, tasks in sorted(filtered_models.items())
        }

        return sorted_filtered_models

    def set_models(self, models_list: List[Dict]):
        self._models = models_list
        filtered_models = self._filter_and_sort_models(self._models)
        self._table_data = filtered_models
        self._model_architectures = list(filtered_models.keys())
        self.__default_selected_arch_type = (
            self._model_architectures[0] if len(self._model_architectures) > 0 else None
        )
        self.__default_selected_task_type = (
            list(filtered_models[self.__default_selected_arch_type].keys())[0]
            if self.__default_selected_arch_type is not None
            else None
        )
        DataJson()[self.widget_id]["tableData"] = self._table_data
        DataJson().send_changes()

        StateJson()[self.widget_id]["selectedRow"] = 0
        StateJson()[self.widget_id]["selectedTaskType"] = self.__default_selected_task_type
        StateJson()[self.widget_id]["selectedArchType"] = self.__default_selected_arch_type
        StateJson().send_changes()

    def arch_type_changed(self, func):
        route_path = self.get_route_path(PretrainedModelsSelector.Routes.ARCH_TYPE_CHANGED)
        server = self._sly_app.get_server()
        self._arch_changes_handled = True

        @server.post(route_path)
        async def _arch_type_changed():
            res = self.get_selected_arch_type()
            func(res)

        return _arch_type_changed

    def task_type_changed(self, func):
        route_path = self.get_route_path(PretrainedModelsSelector.Routes.TASK_TYPE_CHANGED)
        server = self._sly_app.get_server()
        self._task_changes_handled = True

        @server.post(route_path)
        def _task_type_changed():
            res = self.get_selected_task_type()
            func(res)

        return _task_type_changed

    def model_changed(self, func):
        route_path = self.get_route_path(PretrainedModelsSelector.Routes.MODEL_CHANGED)
        server = self._sly_app.get_server()
        self._model_changes_handled = True

        @server.post(route_path)
        def _model_changed():
            res = self.get_selected_row()
            func(res)

        return _model_changed
