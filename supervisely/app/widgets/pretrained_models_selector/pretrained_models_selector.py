from typing import Dict, List, Literal, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget
from supervisely.io.fs import get_file_ext
from supervisely.nn.utils import ModelSource, _get_model_name


class PretrainedModelsSelector(Widget):
    """Widget for selecting pretrained models from a list of models. Groups models by architecture and task type.

    :param models_list: List of models to display in the widget. See usage example for the expected format.
    :type models_list: List[Dict]
    :param widget_id: Unique identifier for the widget. If not provided, a unique ID will be generated.
    :type widget_id: str, optional
    :param sort_models: Whether to sort the task types within each architecture by name. Default is False.
    :type sort_models: bool, optional

    Usage example:

    .. code-block:: python

        from supervisely.app.widgets import PretrainedModelsSelector

        models_list = [
            {
                "Model": "YOLOv8n-det",
                "Size (pixels)": "640",
                "mAP": "37.3",
                "params (M)": "3.2",
                "FLOPs (B)": "8.7",
                "meta": {
                    "taskType": "object detection",
                    "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8n.pt",
                },
            },
            {
                "Model": "YOLOv8s-det",
                "Size (pixels)": "640",
                "mAP": "44.9",
                "params (M)": "11.2",
                "FLOPs (B)": "28.6",
                "meta": {
                    "taskType": "object detection",
                    "weightsURL": "https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8s.pt",
                },
            },
        ]

        pretrained_models_selector = PretrainedModelsSelector(models_list=models_list, sort_models=False)

    """

    class Routes:
        ARCH_TYPE_CHANGED = "arch_type_changed"
        TASK_TYPE_CHANGED = "task_type_changed"
        MODEL_CHANGED = "model_changed"

    def __init__(
        self,
        models_list: List[Dict],
        widget_id: str = None,
        sort_models: bool = False,
    ):
        self._api = Api.from_env()

        self._models = models_list
        filtered_models = self._filter_and_sort_models(self._models, sort_models)

        self._table_data = filtered_models
        self._task_types = self._filter_task_types(list(filtered_models.keys()))
        self._arch_types = []
        # maintain correct order of arch types
        for task_type in self._task_types:
            for arch_type in filtered_models[task_type].keys():
                if arch_type not in self._arch_types:
                    self._arch_types.append(arch_type)

        self._arch_changes_handled = False
        self._task_changes_handled = False
        self._model_changes_handled = False

        self.__default_selected_arch_type = (
            self._arch_types[0] if len(self._arch_types) > 0 else None
        )
        self.__default_selected_task_type = (
            self._task_types[0] if len(self._task_types) > 0 else None
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
            "taskTypes": self._task_types,
            "archTypes": self._arch_types,
            "selectedTaskType": self.__default_selected_task_type,
            "selectedArchType": self.__default_selected_arch_type,
        }

    def get_available_task_types(self) -> List[str]:
        return StateJson()[self.widget_id]["taskTypes"]

    def get_available_arch_types(self) -> List[str]:
        return StateJson()[self.widget_id]["archTypes"]

    def get_selected_task_type(self) -> str:
        return StateJson()[self.widget_id]["selectedTaskType"]

    def get_selected_arch_type(self) -> str:
        return StateJson()[self.widget_id]["selectedArchType"]

    def get_selected_row(self, state=StateJson()) -> Union[Dict, None]:
        task_type = self.get_selected_task_type()
        arch_type = self.get_selected_arch_type()
        if task_type is None or arch_type is None:
            return

        models = self._table_data[task_type][arch_type]
        if len(models) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = int(widget_actual_state["selectedRow"])
            return models[selected_row_index]

    def get_selected_model_params(
        self,
        model_name_column: str = "Model",
        train_version: Literal["v1", "v2"] = "v1",
    ) -> Union[Dict, None]:
        selected_model = self.get_selected_row()
        if selected_model is None:
            return {}

        if train_version == "v1":
            model_name = selected_model.get(model_name_column)
            if model_name is None:
                model_name = _get_model_name(selected_model)
                if model_name is None:
                    raise ValueError(
                        "Could not find model name. Make sure you have column 'Model' in your models list."
                    )

            model_meta = selected_model.get("meta")
            if model_meta is None:
                raise ValueError(
                    "Could not find model meta. Make sure you have key 'meta' in your models configuration list."
                )
            checkpoint_url = model_meta.get("weights_url")
            if checkpoint_url is None:
                model_files = model_meta.get("model_files")
                if model_files is None:
                    raise ValueError(
                        "Could not find model files. Make sure you have key 'model_files' or 'weights_url' in 'meta' in your models configuration list."
                    )
                checkpoint_url = model_files.get("checkpoint")
                if checkpoint_url is None:
                    raise ValueError(
                        "Could not find checkpoint url. Make sure you have key 'checkpoint' in 'model_files' in 'meta' in your models configuration list."
                    )

            checkpoint_ext = get_file_ext(checkpoint_url)
            checkpoint_name = f"{model_name.lower()}{checkpoint_ext}"

            task_type = self.get_selected_task_type()
            model_params = {
                "model_source": "Pretrained models",
                "task_type": task_type,
                "checkpoint_name": checkpoint_name,
                "checkpoint_url": checkpoint_url,
            }

            if len(self._arch_types) > 1:
                arch_type = self.get_selected_arch_type()
                model_params["arch_type"] = arch_type

            config_url = selected_model.get("meta", {}).get("config_url")
            if config_url is not None:
                model_params["config_url"] = config_url
        elif train_version == "v2":
            model_info = self.get_selected_row()
            meta = model_info.get("meta")
            if meta is None:
                raise ValueError("key 'meta' not found in model configuration")
            model_files = meta.get("model_files")
            if model_files is None:
                raise ValueError("key 'model_files' not found in key 'meta' in model configuration")
            model_params = {
                "model_source": ModelSource.PRETRAINED,
                "model_info": model_info,
                "model_files": model_files,
            }
        return model_params

    def get_selected_row_index(self, state=StateJson()) -> Union[int, None]:
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

    def set_active_arch_type(self, arch_type: str) -> None:
        if arch_type not in self._arch_types:
            raise ValueError(f'Architecture type "{arch_type}" does not exist')
        StateJson()[self.widget_id]["selectedArchType"] = arch_type
        StateJson().send_changes()

    def set_active_task_type(self, task_type: str) -> None:
        if task_type not in self._task_types:
            raise ValueError(f'Task type "{task_type}" does not exist')
        StateJson()[self.widget_id]["selectedTaskType"] = task_type
        StateJson().send_changes()

    def set_active_row(self, row_index: int) -> None:
        if row_index < 0:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()

    def set_by_model_name(self, model_name: str) -> None:
        for task_type in self._table_data:
            for arch_type in self._table_data[task_type]:
                for idx, model in enumerate(self._table_data[task_type][arch_type]):
                    name_from_info = _get_model_name(model)
                    if name_from_info is not None:
                        if name_from_info.lower() == model_name.lower():
                            self.set_active_task_type(task_type)
                            self.set_active_arch_type(arch_type)
                            self.set_active_row(idx)
                            return

    def get_by_model_name(self, model_name: str) -> Union[Dict, None]:
        for task_type in self._table_data:
            for arch_type in self._table_data[task_type]:
                for idx, model in enumerate(self._table_data[task_type][arch_type]):
                    name_from_info = _get_model_name(model)
                    if name_from_info is not None:
                        if name_from_info.lower() == model_name.lower():
                            return model

    def _filter_and_sort_models(self, models: List[Dict], sort_models: bool = True) -> Dict:
        filtered_models = {}
        for model in models:
            for key in model:
                if isinstance(model[key], (int, float)):
                    model[key] = str(model[key])

            arch_type = model.get("meta", {}).get("arch_type", "other")
            task_type = model.get("meta", {}).get("task_type", model.get("task_type", "other"))

            if task_type not in filtered_models:
                filtered_models[task_type] = {}
            if arch_type not in filtered_models[task_type]:
                filtered_models[task_type][arch_type] = []
            filtered_models[task_type][arch_type].append(model)

        if sort_models:
            sorted_filtered_models = {
                task: {arch: models for arch, models in sorted(archs.items())}
                for task, archs in sorted(filtered_models.items())
            }
        else:
            sorted_filtered_models = {
                task: {arch: models for arch, models in archs.items()}
                for task, archs in filtered_models.items()
            }
        return sorted_filtered_models

    def _filter_task_types(self, task_types: List[str]):
        order = ["object detection", "instance segmentation", "pose estimation"]
        sorted_tt = [task for task in order if task in task_types]
        other_tasks = sorted(set(task_types) - set(order))
        sorted_tt.extend(other_tasks)
        return sorted_tt

    def set_models(self, models_list: List[Dict], sort_models: bool = False):
        self._models = models_list
        filtered_models = self._filter_and_sort_models(self._models, sort_models)
        self._table_data = filtered_models

        self._task_types = self._filter_task_types(list(filtered_models.keys()))
        self._arch_types = []
        # maintain correct order of arch types
        for task_type in self._task_types:
            for arch_type in filtered_models[task_type].keys():
                if arch_type not in self._arch_types:
                    self._arch_types.append(arch_type)

        self.__default_selected_arch_type = (
            self._arch_types[0] if len(self._arch_types) > 0 else None
        )
        self.__default_selected_task_type = (
            list(filtered_models[self.__default_selected_arch_type].keys())[0]
            if self.__default_selected_arch_type is not None
            else None
        )
        DataJson()[self.widget_id]["tableData"] = self._table_data
        DataJson().send_changes()

        StateJson()[self.widget_id]["selectedRow"] = 0
        StateJson()[self.widget_id]["taskTypes"] = self._task_types
        StateJson()[self.widget_id]["archTypes"] = self._arch_types
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
