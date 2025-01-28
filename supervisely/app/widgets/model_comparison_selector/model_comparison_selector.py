import os
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

from supervisely import env, logger
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api, ApiField
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

COL_CHECKBOX = "".upper()
COL_ID = "task id".upper()
COL_MODEL = "model".upper()
COL_SOURCE = "model source".upper()
COL_CHECKPOINT = "checkpoint".upper()
COL_REPORT = "report".upper()

columns = [COL_CHECKBOX, COL_ID, COL_MODEL, COL_SOURCE, COL_CHECKPOINT, COL_REPORT]


class ModelComparisonSelector(Widget):
    class Routes:
        PROJECT_CHANGED = "project_changed"
        VALUE_CHANGED = "value_changed"

    class ModelRow:
        def __init__(
            self,
            api: Api,
            team_id: int,
            becnhmark_info: dict,
        ):
            self._api = api
            self._team_id = team_id
            self._benchmark_info = becnhmark_info
            self._eval_dir = becnhmark_info["eval_dir"]

            task_id = self._benchmark_info["task_id"]
            if task_id == "debug-session":
                pass
            elif type(task_id) is str:
                if task_id.isdigit():
                    task_id = int(task_id)
                else:
                    raise ValueError(f"Task ID: '{task_id}' is not a number")

            # col 1 task
            self._task_id = task_id
            self._task_type = self._benchmark_info["task_type"]
            self._report_path = self._benchmark_info["report_path"]
            self._report_date = self._benchmark_info["created_at"]
            self._report_link = self._create_tf_report_link()

            # col 2 model
            self._model_name = self._benchmark_info["model_name"]

            # col 3 model source
            self._model_source = self._benchmark_info["model_source"]

            # col 4 checkpoint
            self._checkpoint_name = self._benchmark_info["checkpoint_name"]
            checkpoint_link = self._benchmark_info["checkpoint_url"]
            if self._model_source == ModelSource.PRETRAINED:
                self._checkpoint_link = checkpoint_link
            else:
                if is_development():
                    self._checkpoint_link = abs_url(f"{checkpoint_link}")
                else:
                    self._checkpoint_link = checkpoint_link

            # col 5 benchmark report
            self._report_id = self._benchmark_info["report_id"]

            # ----------------------------- #
            self._gt_project_id = self._benchmark_info["gt_project_id"]

            # widgets
            self._task_widget = self._create_tf_report_widget()
            self._model_widget = self._create_model_widget()
            self._model_source_widget = self._create_model_source_widget()
            self._checkpoints_widget = self._create_checkpoint_widget()
            self._benchmark_widget = self._create_benchmark_widget()

        @property
        def task_id(self) -> int:
            return self._task_id

        @property
        def task_type(self) -> str:
            return self._task_type

        @property
        def report_date(self) -> str:
            return self._report_date

        @property
        def report_id(self) -> str:
            return self._report_id

        @property
        def report_link(self) -> str:
            return self._report_link

        @property
        def gt_project_id(self) -> ProjectInfo:
            return self._gt_project_id

        @property
        def eval_dir(self) -> str:
            return self._eval_dir

        @property
        def model_name(self) -> str:
            return self._model_name

        @property
        def model_source(self) -> str:
            return self._model_source

        @property
        def checkpoint_name(self) -> str:
            return self._checkpoint_name

        @property
        def checkpoint_link(self) -> str:
            return self._checkpoint_link

        def to_html(self) -> List[str]:
            return [
                f"<div> {self._task_widget.to_html()} </div>",
                f"<div> {self._model_widget.to_html()} </div>",
                f"<div> {self._model_source_widget.to_html()} </div>",
                f"<div> {self._checkpoints_widget.to_html()} </div>",
                f"<div> {self._benchmark_widget.to_html()} </div>",
            ]

        def _create_tf_report_link(self) -> str:
            report_file = self._api.file.get_info_by_path(self._team_id, self._report_path)
            if report_file is not None:
                if is_development():
                    return abs_url(f"/files/{report_file.id}")
                else:
                    return f"/files/{report_file.id}"
            else:
                return ""

        def _generate_session_link(self) -> str:
            if is_development():
                session_link = abs_url(f"/apps/sessions/{self._task_id}")
            else:
                session_link = f"/apps/sessions/{self._task_id}"
            return session_link

        def _create_tf_report_widget(self) -> Flexbox:
            task_widget = Container(
                [
                    Text(
                        f"<i class='zmdi zmdi-folder' style='color: #7f858e'></i> <a href='{self._report_link}' target='_blank'>{self._task_id}</a>",
                        "text",
                    ),
                    Text(
                        f"<span class='field-description text-muted' style='color: #7f858e'>{self._report_date}</span>",
                        "text",
                        font_size=13,
                    ),
                ],
                gap=0,
            )
            return task_widget

        def _create_model_widget(self) -> Text:
            model_widget = Text(
                f"<span class='field-description text-muted' style='color: #7f858e'>{self._model_name}</span>",
                "text",
                font_size=13,
            )
            return model_widget

        def _create_model_source_widget(self) -> Text:
            model_source_widget = Text(
                f"<span class='field-description text-muted' style='color: #7f858e'>{self._model_source}</span>",
                "text",
                font_size=13,
            )
            return model_source_widget

        def _create_checkpoint_widget(self) -> Select:
            checkpoint_widget = Text(
                f"<i class='zmdi zmdi-file' style='color: #7f858e'></i> <a href='{self._checkpoint_link}' target='_blank'>{self._checkpoint_name}</a>",
                "text",
            )
            return checkpoint_widget

        def _create_benchmark_widget(self) -> Text:
            if is_development():
                benchmark_report_link = abs_url(f"/model-benchmark?id={self._report_id}")
            else:
                benchmark_report_link = f"/model-benchmark?id={self._report_id}"

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

        benchmark_infos = self._parse_benchmark_infos()

        self.__debug_row = None

        with ThreadPoolExecutor() as executor:
            future = executor.submit(self._generate_table_rows, benchmark_infos)
            table_rows = future.result()

        self._columns = columns
        self._rows = table_rows
        # self._rows_html = #[row.to_html() for row in self._rows]

        self._project_ids = [project_id for project_id in table_rows]
        self._rows_html = defaultdict(list)
        for project_id in table_rows:
            self._rows_html[project_id].extend(
                [model_row.to_html() for model_row in table_rows[project_id]]
            )

        self._changes_handled = False
        self._project_changes_handled = False
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
            "projectIds": self._project_ids,
        }

    def get_json_state(self) -> Dict:
        return {
            "selectedRows": [],
            "selectedProjectId": self._project_ids[0],
        }

    def set_active_project_id(self, project_id: str):
        if project_id not in self._project_ids:
            raise ValueError(f'Project ID: "{project_id}" does not exist')
        StateJson()[self.widget_id]["selectedProjectId"] = project_id
        StateJson().send_changes()

    def get_available_project_ids(self) -> List[str]:
        return self._project_ids

    def disable_table(self) -> None:
        for project_id in self._rows:
            for row in self._rows[project_id]:
                row.checkpoints_selector.disable()
        super().disable()

    def enable_table(self) -> None:
        for project_id in self._rows:
            for row in self._rows[project_id]:
                row.checkpoints_selector.enable()
        super().enable()

    def enable(self):
        self.enable_table()
        super().enable()

    def disable(self) -> None:
        self.disable_table()
        super().disable()

    def _generate_table_rows(self, benchmark_infos: List[dict]) -> Dict[str, List[ModelRow]]:
        """Method to generate table rows from remote path to training app save directory"""

        def process_experiment_info(becnhmark_info: ExperimentInfo):
            try:
                model_row = ModelComparisonSelector.ModelRow(
                    api=self._api,
                    team_id=self._team_id,
                    becnhmark_info=becnhmark_info,
                )
                return becnhmark_info["src_project_id"], model_row.task_type, model_row
            except Exception as e:
                logger.warning(f"Failed to process benchmark info. Error: '{repr(e)}'")
                return None, None

        # table_rows = defaultdict(list)
        table_rows = defaultdict(lambda: defaultdict(list))
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_experiment_info, benchmark_info): benchmark_info
                for benchmark_info in benchmark_infos
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    project_id, task_type, model_row = result
                    if project_id is not None and model_row is not None:
                        if model_row.task_id == "debug-session":
                            self.__debug_row = (project_id, model_row.task_type, model_row)
                            continue
                        table_rows[project_id][task_type].append(model_row)
        self._sort_table_rows(table_rows)
        if self.__debug_row and is_development():
            project_id, model_row.task_type, model_row = self.__debug_row
            table_rows[project_id][model_row.task_type].insert(0, model_row)
        return table_rows

    def _sort_table_rows(self, table_rows: Dict[str, List[ModelRow]]) -> None:
        for project_id in table_rows:
            for task_type in table_rows[project_id]:
                table_rows[project_id][task_type].sort(
                    key=lambda row: int(row.task_id), reverse=True
                )

    def get_selected_rows(self, state=StateJson()) -> Union[ModelRow, None]:
        if len(self._rows) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        project_id = widget_actual_state["selectedProjectId"]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_indexes = widget_actual_state["selectedRows"]

            rows = []
            for i in selected_row_indexes:
                rows.append(self._rows[project_id][i])
            return rows

    def get_selected_row_indexes(self, state=StateJson()) -> Union[int, None]:
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRows"]

    def get_selected_project_id(self) -> str:
        return StateJson()[self.widget_id]["selectedProjectId"]

    def get_selected_experiment_info(self) -> Dict[str, Any]:
        if len(self._rows) == 0:
            return
        selected_row = self.get_selected_row()
        selected_row_json = asdict(selected_row._experiment_info)
        return selected_row_json

    def set_active_row(self, row_indexes: List[int]) -> None:
        indexes_to_set = []
        for row_index in row_indexes:
            if row_index < 0 or row_index > len(self._rows) - 1:
                raise ValueError(f'Row with index "{row_index}" does not exist')
            indexes_to_set.append(row_index)
        StateJson()[self.widget_id]["selectedRows"] = indexes_to_set
        StateJson().send_changes()

    def set_by_task_id(self, task_id: int) -> None:
        for project_id in self._rows:
            for i, row in enumerate(self._rows[project_id]):
                if row.task_id == task_id:
                    self.set_active_row(i)
                    return

    def get_by_task_id(self, task_id: int) -> Union[ModelRow, None]:
        for project_id in self._rows:
            for row in self._rows[project_id]:
                if row.task_id == task_id:
                    return row
        return None

    def project_changed(self, func: Callable):
        route_path = self.get_route_path(ModelComparisonSelector.Routes.PROJECT_CHANGED)
        server = self._sly_app.get_server()
        self._project_changes_handled = True

        @server.post(route_path)
        def _project_changed():
            res = self.get_selected_project_id()
            func(res)

        return _project_changed

    def value_changed(self, func: Callable):
        route_path = self.get_route_path(ModelComparisonSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_rows()
            func(res)

        return _value_changed

    def _parse_benchmark_infos(self) -> List[dict]:
        reports_folder = "/model-benchmark/"
        report_file_name = "inference_info.json"

        report_paths = []
        all_files = self._api.file.list(self._team_id, reports_folder, True, "fileinfo")
        for file in all_files:
            if file.name == report_file_name:
                report_paths.append(file.path)

            def fetch_report_data(report_path: str) -> None:
                try:
                    path_parts = report_path.split("/")
                    project_part = path_parts[2]
                    task_part = path_parts[3]

                    project_id = project_part.split("_")[0]
                    task_id = task_part.split("_")[0]

                    response = self._api.post(
                        "file-storage.download",
                        {ApiField.TEAM_ID: self._team_id, ApiField.PATH: report_path},
                        stream=True,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    response_json["src_project_id"] = project_id
                    response_json["task_id"] = task_id
                    response_json["report_path"] = report_path
                    response_json["created_at"] = file.created_at
                    response_json["eval_dir"] = f"/model-benchmark/{project_part}/{task_part}/"

                    # Get report vue file id
                    report_vue_path = f"{response_json['eval_dir']}visualizations/template.vue"
                    report_file_info = self._api.file.get_info_by_path(
                        self._team_id, report_vue_path
                    )
                    if report_file_info is not None:
                        response_json["report_id"] = report_file_info.id
                    else:
                        response_json["report_id"] = None
                    return response_json
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request failed for '{report_path}': {e}")
                except JSONDecodeError as e:
                    logger.debug(f"JSON decode failed for '{report_path}': {e}")
                except TypeError as e:
                    logger.error(f"TypeError for '{report_path}': {e}")

        report_infos = []
        with ThreadPoolExecutor() as executor:
            report_infos = list(executor.map(fetch_report_data, report_paths))
        return report_infos
