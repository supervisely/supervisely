import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

from supervisely import env, logger
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Container,
    Flexbox,
    ProjectThumbnail,
    Select,
    Text,
    Widget,
)
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.experiments import ExperimentInfo
from supervisely.nn.utils import ModelSource


WEIGHTS_DIR = "weights"

COL_ID = "task id".upper()
COL_MODEL = "model".upper()
COL_PROJECT = "training data".upper()
COL_CHECKPOINTS = "checkpoints".upper()
COL_SESSION = "session".upper()
COL_BENCHMARK = "benchmark".upper()

columns = [COL_ID, COL_MODEL, COL_PROJECT, COL_CHECKPOINTS, COL_SESSION, COL_BENCHMARK]


class ExperimentSelector(Widget):
    class Routes:
        TASK_TYPE_CHANGED = "task_type_changed"
        VALUE_CHANGED = "value_changed"

    class ModelRow:
        def __init__(
            self,
            api: Api,
            team_id: int,
            task_type: str,
            experiment_info: ExperimentInfo,
        ):
            self._api = api
            self._team_id = team_id
            self._task_type = task_type
            self._experiment_info = experiment_info

            task_id = experiment_info.task_id
            if task_id == "debug-session":
                pass
            elif type(task_id) is str:
                if task_id.isdigit():
                    task_id = int(task_id)
                else:
                    raise ValueError(f"Task id {task_id} is not a number")

            # col 1 task
            self._task_id = task_id
            self._task_path = experiment_info.artifacts_dir
            self._task_date = experiment_info.datetime
            self._task_link = self._create_task_link()
            self._config_path = experiment_info.model_files.get("config")
            if self._config_path is not None:
                self._config_path = os.path.join(experiment_info.artifacts_dir, self._config_path)

            # col 2 model
            self._model_name = experiment_info.model_name

            # col 3 project
            self._training_project_id = experiment_info.project_id
            if self._training_project_id is None:
                self._training_project_info = None
            else:
                self._training_project_info = self._api.project.get_info_by_id(
                    self._training_project_id
                )

            # col 4 checkpoints
            self._checkpoints = experiment_info.checkpoints

            self._checkpoints_names = []
            self._checkpoints_paths = []
            self._best_checkpoint_value = None
            for checkpoint_path in self._checkpoints:
                self._checkpoints_names.append(get_file_name_with_ext(checkpoint_path))
                self._checkpoints_paths.append(
                    os.path.join(experiment_info.artifacts_dir, checkpoint_path)
                )
                if experiment_info.best_checkpoint == get_file_name_with_ext(checkpoint_path):
                    self._best_checkpoint = os.path.join(
                        experiment_info.artifacts_dir, checkpoint_path
                    )

            # col 5 session
            self._session_link = self._generate_session_link()

            # col 6 benchmark report
            self._benchmark_report_id = experiment_info.evaluation_report_id

            # widgets
            self._task_widget = self._create_task_widget()
            self._model_wiget = self._create_model_widget()
            self._training_project_widget = self._create_training_project_widget()
            self._checkpoints_widget = self._create_checkpoints_widget()
            self._session_widget = self._create_session_widget()
            self._benchmark_widget = self._create_benchmark_widget()

        @property
        def task_id(self) -> int:
            return self._task_id

        @property
        def task_date(self) -> str:
            return self._task_date

        @property
        def task_link(self) -> str:
            return self._task_link

        @property
        def task_type(self) -> str:
            return self._task_type

        @property
        def training_project_info(self) -> ProjectInfo:
            return self._training_project_info

        @property
        def checkpoints_names(self) -> List[str]:
            return self._checkpoints_names

        @property
        def checkpoints_paths(self) -> List[str]:
            return self._checkpoints_paths

        @property
        def checkpoints_selector(self) -> Select:
            return self._checkpoints_widget

        @property
        def session_link(self) -> str:
            return self._session_link

        @property
        def config_path(self) -> str:
            return self._config_path

        def get_selected_checkpoint_path(self) -> str:
            return self._checkpoints_widget.get_value()

        def get_selected_checkpoint_name(self) -> str:
            return self._checkpoints_widget.get_label()

        def set_selected_checkpoint_by_name(self, checkpoint_name: str):
            for i, name in enumerate(self._checkpoints_names):
                if name == checkpoint_name:
                    self._checkpoints_widget.set_value(self._checkpoints_paths[i])
                    return

        def set_selected_checkpoint_by_path(self, checkpoint_path: str):
            for i, path in enumerate(self._checkpoints_paths):
                if path == checkpoint_path:
                    self._checkpoints_widget.set_value(path)
                    return

        def to_html(self) -> List[str]:
            return [
                f"<div> {self._task_widget.to_html()} </div>",
                f"<div> {self._model_wiget.to_html()} </div>",
                f"<div> {self._training_project_widget.to_html()} </div>",
                f"<div> {self._checkpoints_widget.to_html()} </div>",
                f"<div> {self._session_widget.to_html()} </div>",
                f"<div> {self._benchmark_widget.to_html()} </div>",
            ]

        def _create_task_link(self) -> str:
            remote_path = os.path.join(self._task_path, "open_app.lnk")
            task_file = self._api.file.get_info_by_path(self._team_id, remote_path)
            if task_file is not None:
                if is_development():
                    return abs_url(f"/files/{task_file.id}")
                else:
                    return f"/files/{task_file.id}"
            else:
                return ""

        def _generate_session_link(self) -> str:
            if is_development():
                session_link = abs_url(f"/apps/sessions/{self._task_id}")
            else:
                session_link = f"/apps/sessions/{self._task_id}"
            return session_link

        def _create_task_widget(self) -> Flexbox:
            task_widget = Container(
                [
                    Text(
                        f"<i class='zmdi zmdi-folder' style='color: #7f858e'></i> <a href='{self._task_link}' target='_blank'>{self._task_id}</a>",
                        "text",
                    ),
                    Text(
                        f"<span class='field-description text-muted' style='color: #7f858e'>{self._task_date}</span>",
                        "text",
                        font_size=13,
                    ),
                ],
                gap=0,
            )
            return task_widget

        def _create_model_widget(self) -> Text:
            if self._model_name is None:
                self._model_name = "Unknown model"

            model_widget = Text(
                f"<span class='field-description text-muted' style='color: #7f858e'>{self._model_name}</span>",
                "text",
                font_size=13,
            )
            return model_widget

        def _create_training_project_widget(self) -> Union[ProjectThumbnail, Text]:
            if self.training_project_info is not None:
                training_project_widget = ProjectThumbnail(
                    self._training_project_info, remove_margins=True
                )
            else:
                training_project_widget = Text(
                    f"<span class='field-description text-muted' style='color: #7f858e'>Project was deleted</span>",
                    "text",
                    font_size=13,
                )
            return training_project_widget

        def _create_checkpoints_widget(self) -> Select:
            checkpoint_selector_items = []
            for path, name in zip(self._checkpoints_paths, self._checkpoints_names):
                checkpoint_selector_items.append(Select.Item(value=path, label=name))
            checkpoint_selector = Select(items=checkpoint_selector_items)
            if self._best_checkpoint_value is not None:
                checkpoint_selector.set_value(self._best_checkpoint)
            return checkpoint_selector

        def _create_session_widget(self) -> Text:
            session_link_widget = Text(
                f"<a href='{self._session_link}' target='_blank'>Preview</a> <i class='zmdi zmdi-open-in-new'></i>",
                "text",
            )
            return session_link_widget

        def _create_benchmark_widget(self) -> Text:
            if self._benchmark_report_id is None:
                self._benchmark_report_id = "No evaluation report available"
                benchmark_widget = Text(
                    "<span class='field-description text-muted' style='color: #7f858e'>No evaluation report available</span>",
                    "text",
                    font_size=13,
                )
            else:
                if is_development():
                    benchmark_report_link = abs_url(
                        f"/model-benchmark?id={self._benchmark_report_id}"
                    )
                else:
                    benchmark_report_link = f"/model-benchmark?id={self._benchmark_report_id}"

                benchmark_widget = Text(
                    f"<i class='zmdi zmdi-chart' style='color: #7f858e'></i> <a href='{benchmark_report_link}' target='_blank'>evaluation report</a>",
                    "text",
                )
            return benchmark_widget

    def __init__(
        self,
        team_id: int,
        experiment_infos: List[ExperimentInfo] = [],
        widget_id: str = None,
    ):
        self._api = Api.from_env()

        self._team_id = team_id
        self.__debug_row = None

        with ThreadPoolExecutor() as executor:
            future = executor.submit(self._generate_table_rows, experiment_infos)
            table_rows = future.result()

        self._columns = columns
        self._rows = table_rows
        # self._rows_html = #[row.to_html() for row in self._rows]

        task_types = [task_type for task_type in table_rows]
        self._rows_html = defaultdict(list)
        for task_type in table_rows:
            self._rows_html[task_type].extend(
                [model_row.to_html() for model_row in table_rows[task_type]]
            )

        self._task_types = self._filter_task_types(task_types)
        if len(self._task_types) == 0:
            self.__default_selected_task_type = None
        else:
            self.__default_selected_task_type = self._task_types[0]

        self._changes_handled = False
        self._task_type_changes_handled = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def rows(self) -> Dict[str, List[ModelRow]]:
        return self._rows

    def get_json_data(self) -> Dict:
        return {
            "columns": self._columns,
            "rowsHtml": self._rows_html,
            "taskTypes": self._task_types,
        }

    def get_json_state(self) -> Dict:
        return {
            "selectedRow": 0,
            "selectedTaskType": self.__default_selected_task_type,
        }

    def set_active_task_type(self, task_type: str):
        if task_type not in self._task_types:
            raise ValueError(f'Task Type "{task_type}" does not exist')
        StateJson()[self.widget_id]["selectedTaskType"] = task_type
        StateJson().send_changes()

    def get_available_task_types(self) -> List[str]:
        return self._task_types

    def disable_table(self) -> None:
        for task_type in self._rows:
            for row in self._rows[task_type]:
                row.checkpoints_selector.disable()
        super().disable()

    def enable_table(self) -> None:
        for task_type in self._rows:
            for row in self._rows[task_type]:
                row.checkpoints_selector.enable()
        super().enable()

    def enable(self):
        self.enable_table()
        super().enable()

    def disable(self) -> None:
        self.disable_table()
        super().disable()

    def _generate_table_rows(
        self, experiment_infos: List[ExperimentInfo]
    ) -> Dict[str, List[ModelRow]]:
        """Method to generate table rows from remote path to training app save directory"""

        def process_experiment_info(experiment_info: ExperimentInfo):
            try:
                model_row = ExperimentSelector.ModelRow(
                    api=self._api,
                    team_id=self._team_id,
                    task_type=experiment_info.task_type,
                    experiment_info=experiment_info,
                )
                return experiment_info.task_type, model_row
            except Exception as e:
                logger.warn(f"Failed to process experiment info: {experiment_info}")
                return None, None

        table_rows = defaultdict(list)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_experiment_info, experiment_info): experiment_info
                for experiment_info in experiment_infos
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    task_type, model_row = result
                    if task_type is not None and model_row is not None:
                        if model_row.task_id == "debug-session":
                            self.__debug_row = (task_type, model_row)
                            continue
                        table_rows[task_type].append(model_row)
        self._sort_table_rows(table_rows)
        if self.__debug_row and is_development():
            task_type, model_row = self.__debug_row
            table_rows[task_type].insert(0, model_row)
        return table_rows

    def _sort_table_rows(self, table_rows: Dict[str, List[ModelRow]]) -> None:
        for task_type in table_rows:
            table_rows[task_type].sort(key=lambda row: int(row.task_id), reverse=True)

    def _filter_task_types(self, task_types: List[str]):
        sorted_tt = []
        if "object detection" in task_types:
            sorted_tt.append("object detection")
        if "instance segmentation" in task_types:
            sorted_tt.append("instance segmentation")
        if "pose estimation" in task_types:
            sorted_tt.append("pose estimation")
        other_tasks = sorted(
            set(task_types)
            - set(
                [
                    "object detection",
                    "instance segmentation",
                    "semantic segmentation",
                    "pose estimation",
                ]
            )
        )
        sorted_tt.extend(other_tasks)
        return sorted_tt

    def get_selected_row(self, state=StateJson()) -> Union[ModelRow, None]:
        if len(self._rows) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        task_type = widget_actual_state["selectedTaskType"]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = int(widget_actual_state["selectedRow"])
            return self._rows[task_type][selected_row_index]

    def get_selected_row_index(self, state=StateJson()) -> Union[int, None]:
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

    def get_selected_task_type(self) -> str:
        return StateJson()[self.widget_id]["selectedTaskType"]

    def get_selected_experiment_info(self) -> Dict[str, Any]:
        if len(self._rows) == 0:
            return
        selected_row = self.get_selected_row()
        selected_row_json = asdict(selected_row._experiment_info)
        return selected_row_json

    def get_selected_checkpoint_path(self) -> str:
        if len(self._rows) == 0:
            return
        selected_row = self.get_selected_row()
        return selected_row.get_selected_checkpoint_path()

    def get_model_files(self) -> Dict[str, str]:
        """
        Returns a dictionary with full paths to model files in Supervisely Team Files.
        """
        experiment_info = self.get_selected_experiment_info()
        artifacts_dir = experiment_info.get("artifacts_dir")
        model_files = experiment_info.get("model_files", {})

        full_model_files = {
            name: os.path.join(artifacts_dir, file) for name, file in model_files.items()
        }
        full_model_files["checkpoint"] = self.get_selected_checkpoint_path()
        return full_model_files

    def get_deploy_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary with deploy parameters except runtime and device keys.
        """
        deploy_params = {
            "model_source": ModelSource.CUSTOM,
            "model_files": self.get_model_files(),
            "model_info": self.get_selected_experiment_info(),
        }
        return deploy_params

    def set_active_row(self, row_index: int) -> None:
        if row_index < 0 or row_index > len(self._rows) - 1:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()

    def set_by_task_id(self, task_id: int) -> None:
        for task_type in self._rows:
            for i, row in enumerate(self._rows[task_type]):
                if row.task_id == task_id:
                    self.set_active_row(i)
                    return

    def get_by_task_id(self, task_id: int) -> Union[ModelRow, None]:
        for task_type in self._rows:
            for row in self._rows[task_type]:
                if row.task_id == task_id:
                    return row
        return None

    def task_type_changed(self, func: Callable):
        route_path = self.get_route_path(ExperimentSelector.Routes.TASK_TYPE_CHANGED)
        server = self._sly_app.get_server()
        self._task_type_changes_handled = True

        @server.post(route_path)
        def _task_type_changed():
            res = self.get_selected_task_type()
            func(res)

        return _task_type_changed

    def value_changed(self, func: Callable):
        route_path = self.get_route_path(ExperimentSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_row()
            func(res)

        return _value_changed
