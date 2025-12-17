import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from supervisely import batched
from supervisely._utils import abs_url, is_development, logger
from supervisely.api.api import Api, ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.app.exceptions import show_dialog
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.dropdown_checkbox_selector.dropdown_checkbox_selector import (
    DropdownCheckboxSelector,
)
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.app.widgets.flexbox.flexbox import Flexbox
from supervisely.app.widgets.project_thumbnail.project_thumbnail import ProjectThumbnail
from supervisely.app.widgets.select.select import Select
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.widget import Widget
from supervisely.io import env
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.experiments import ExperimentInfo


class ExperimentSelector(Widget):
    """
    Widget for selecting experiments from a team.
    """

    class COLUMN:
        NAME = "TASK ID"
        MODEL = "MODEL"
        TRAINING_DATA = "TRAINING DATA"
        CHECKPOINTS = "CHECKPOINTS"
        SESSION = "SESSION"
        BENCHMARK = "BENCHMARK"

    COLUMNS = [
        COLUMN.NAME,
        COLUMN.MODEL,
        COLUMN.TRAINING_DATA,
        COLUMN.CHECKPOINTS,
        COLUMN.SESSION,
        COLUMN.BENCHMARK,
    ]

    class ModelRow:
        def __init__(
            self,
            api: Api,
            team_id: int,
            task_type: str,
            experiment_info: ExperimentInfo,
            project_info: Optional[ProjectInfo] = None,
        ):
            self._api = api
            self._team_id = team_id
            self._task_type = task_type
            self._experiment_info = experiment_info
            self._project_info = project_info

            task_id = experiment_info.task_id
            if task_id == -1:
                pass
            elif task_id == "debug-session":
                task_id = -1
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
                self._training_project_info = self._project_info

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
            training_project_thumbnail = ProjectThumbnail(
                self._training_project_info, remove_margins=True
            )
            training_project_text = Text(
                f"<span class='field-description text-muted' style='color: #7f858e'>Project was deleted</span>",
                "text",
                font_size=13,
            )
            if self.training_project_info is not None:
                training_project_thumbnail.show()
                training_project_text.hide()
            else:
                training_project_thumbnail.hide()
                training_project_text.show()
            return Container(widgets=[training_project_thumbnail, training_project_text], gap=0)

        def checkpoint_changed(self, checkpoint_value: str):
            return

        def _create_checkpoints_widget(self) -> Select:
            checkpoint_selector_items = []
            for path, name in zip(self._checkpoints_paths, self._checkpoints_names):
                checkpoint_selector_items.append(Select.Item(value=path, label=name))
            checkpoint_selector = Select(items=checkpoint_selector_items)
            if self._best_checkpoint_value is not None:
                checkpoint_selector.set_value(self._best_checkpoint)

            @checkpoint_selector.value_changed
            def on_checkpoint_changed(checkpoint_value: str):
                self.checkpoint_changed(checkpoint_value)

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
                    f"<i class='zmdi zmdi-chart' style='color: #7f858e'></i> <a href='{benchmark_report_link}' target='_blank'>Evaluation report</a>",
                    "text",
                )
            return benchmark_widget

        def _widget_to_cell_value(self, widget: Widget) -> str:
            if isinstance(widget, Container):
                return json.dumps(
                    {
                        "widget_id": widget.widget_id,
                        "widgets": [w.widget_id for w in widget._widgets],
                    }
                )
            else:
                return json.dumps({"widget_id": widget.widget_id, "widgets": []})

        def to_table_row(self):
            return [
                self._widget_to_cell_value(w)
                for w in [
                    self._task_widget,
                    self._model_wiget,
                    self._training_project_widget,
                    self._checkpoints_widget,
                    self._session_widget,
                    self._benchmark_widget,
                ]
            ]

        @classmethod
        def widgets_templates(cls):
            checkpoints_template_widget = Select(items=[])
            checkpoints_template_widget.value_changed(lambda _: None)

            return [
                # _task_widget
                Container(widgets=[Text(""), Text("")], gap=0),
                # _model_wiget
                Text(""),
                # _training_project_widget
                Container(widgets=[ProjectThumbnail(remove_margins=True), Text("")], gap=0),
                # _checkpoints_widget
                checkpoints_template_widget,
                # _session_widget
                Text(""),
                # _benchmark_widget
                Text(""),
            ]

        def search_text(self) -> str:
            text = ""
            text += str(self._task_id)
            text += str(self._task_date)
            text += str(self._model_name)
            if self._training_project_info is not None:
                text += str(self._training_project_info.name)
            else:
                text += "Project was deleted"
            return text

        def sort_values(self) -> List[int]:
            # Sort by training project name: real names first (A->Z), deleted projects last
            if self._training_project_info is not None:
                training_project_name = (0, self._training_project_info.name.lower())
            else:
                training_project_name = (1, "")

            if self._benchmark_report_id == "No evaluation report available":
                benchmark_report_id = 0
            else:
                benchmark_report_id = 1

            return [
                self._task_id,
                self._model_name.capitalize(),
                training_project_name,
                0,
                0,
                benchmark_report_id,
            ]

    def __init__(
        self,
        api: Api = None,
        team_id: int = None,
        experiment_infos: List[ExperimentInfo] = [],
        widget_id: str = None,
    ):
        if team_id is None:
            team_id = env.team_id()
        self.team_id = team_id
        if api is None:
            api = Api()
        self.api = api
        self._experiment_infos = experiment_infos
        self._checkpoint_changed_func = None

        self._rows = []
        self.table = self._create_table()
        self._rows_search_texts = []
        self._rows_sort_values = []

        self._project_infos_map = self._get_project_infos_map(experiment_infos)
        self.set_experiment_infos(experiment_infos)
        super().__init__(widget_id=widget_id)

    def _search_function(self, data: pd.DataFrame, search_value: str) -> List[ModelRow]:
        search_texts = []
        for idx in data.index:
            first_col_value = data.loc[idx, self.COLUMNS[0]]
            if isinstance(first_col_value, pd.Series):
                first_col_value = first_col_value.iloc[0]
            original_idx = self._first_column_value_to_index[first_col_value]
            search_texts.append(self._rows_search_texts[original_idx])

        search_series = pd.Series(search_texts, index=data.index)
        mask = search_series.str.contains(search_value, case=False, na=False)
        return data[mask]

    def _sort_function(
        self, data: pd.DataFrame, column_idx: int, order: str = "asc"
    ) -> List[ModelRow]:
        data = data.copy()
        if column_idx >= len(self._rows_sort_values[0]) if self._rows_sort_values else True:
            raise IndexError(
                f"Sorting by column idx = {column_idx} is not possible, your sort values have only {len(self._rows_sort_values[0]) if self._rows_sort_values else 0} columns with idx from 0 to {len(self._rows_sort_values[0]) - 1 if self._rows_sort_values else -1}"
            )

        if order == "asc":
            ascending = True
        else:
            ascending = False

        try:
            sort_values = []
            for idx in data.index:
                first_col_value = data.loc[idx, self.COLUMNS[0]]
                if isinstance(first_col_value, pd.Series):
                    first_col_value = first_col_value.iloc[0]
                original_idx = self._first_column_value_to_index[first_col_value]
                sort_values.append(self._rows_sort_values[original_idx][column_idx])

            sort_series = pd.Series(sort_values, index=data.index)
            sorted_indices = sort_series.sort_values(ascending=ascending).index
            data = data.loc[sorted_indices]
            data.reset_index(inplace=True, drop=True)

        except IndexError as e:
            e.args = (
                f"Sorting by column idx = {column_idx} is not possible, your sort values have only {len(self._rows_sort_values[0]) if self._rows_sort_values else 0} columns with idx from 0 to {len(self._rows_sort_values[0]) - 1 if self._rows_sort_values else -1}",
            )
            raise e

        return data

    def _filter_function(
        self, data: pd.DataFrame, filter_value: Tuple[List[str], List[str]]
    ) -> pd.DataFrame:
        try:
            frameworks, task_types = filter_value

            filtered_experiments_idxs = set()
            if not frameworks and not task_types:
                return data

            for idx, experiment_info in enumerate(self._experiment_infos):
                should_add = True
                if frameworks and experiment_info.framework_name not in frameworks:
                    should_add = False
                if task_types and experiment_info.task_type not in task_types:
                    should_add = False
                if should_add:
                    filtered_experiments_idxs.add(idx)

            filtered_data = data.iloc[sorted(filtered_experiments_idxs)]
            filtered_data.reset_index(inplace=True, drop=True)
            return filtered_data
        except Exception as e:
            logger.error(f"Error during filtering: {e}", exc_info=True)
            show_dialog(title="Filtering Error", description=str(e), status="error")
            return data

    def _get_frameworks(self):
        frameworks = set()
        for experiment_info in self._experiment_infos:
            frameworks.add(experiment_info.framework_name)
        return sorted(frameworks)

    def _get_task_types(self):
        task_types = set()
        for experiment_info in self._experiment_infos:
            task_types.add(experiment_info.task_type)
        return sorted(task_types)

    def _create_table(self) -> FastTable:
        widgets = self.ModelRow.widgets_templates()
        columns = []
        columns_options = []
        for column_name, widget in zip(self.COLUMNS, widgets):
            columns.append((column_name, widget))
            columns_options.append({"customCell": True})
        columns_options[3].update({"classes": "border border-gray-200 px-2"})
        columns_options[3].update({"disableSort": True})
        columns_options[4].update({"disableSort": True})
        self.framework_filter = DropdownCheckboxSelector(
            label="Framework",
            items=[
                DropdownCheckboxSelector.Item(framework) for framework in self._get_frameworks()
            ],
        )
        self.task_type_filter = DropdownCheckboxSelector(
            label="Task Type",
            items=[
                DropdownCheckboxSelector.Item(task_type) for task_type in self._get_task_types()
            ],
        )
        table = FastTable(
            columns=columns,
            columns_options=columns_options,
            is_radio=True,
            page_size=10,
            header_right_content=Container(
                widgets=[self.framework_filter, self.task_type_filter],
                gap=10,
                direction="horizontal",
            ),
        )
        table.set_search(self._search_function)
        table.set_sort(self._sort_function)
        table.set_filter(self._filter_function)

        @self.framework_filter.value_changed
        def on_framework_filter_change(
            selected_frameworks: List[DropdownCheckboxSelector.Item],
        ):
            selected_frameworks = [item.id for item in selected_frameworks]
            selected_task_types = self.task_type_filter.get_selected()
            self.table.filter((selected_frameworks, selected_task_types))

        @self.task_type_filter.value_changed
        def on_task_type_filter_change(
            selected_task_types: List[DropdownCheckboxSelector.Item],
        ):
            selected_task_types = [item.id for item in selected_task_types]
            selected_frameworks = self.framework_filter.get_selected()
            self.table.filter((selected_frameworks, selected_task_types))

        return table

    def _get_project_infos_map(
        self, experiment_infos: List[ExperimentInfo]
    ) -> Dict[int, ProjectInfo]:
        """
        Returns a map of project IDs to project infos used in the experiment infos.
        """
        project_ids = set()
        for experiment_info in experiment_infos:
            if experiment_info.project_id is not None:
                project_ids.add(experiment_info.project_id)
        project_ids = list(project_ids)

        project_infos_map = {}
        if project_ids is not None:
            for batch in batched(project_ids):
                filters = [
                    {
                        ApiField.FIELD: ApiField.ID,
                        ApiField.OPERATOR: "in",
                        ApiField.VALUE: batch,
                    },
                ]

                fields = [ApiField.IMAGES_COUNT, ApiField.REFERENCE_IMAGE_URL]
                batch_infos = self.api.project.get_list(
                    team_id=self.team_id,
                    filters=filters,
                    fields=fields,
                )
                for info in batch_infos:
                    project_infos_map[info.id] = info

        return project_infos_map

    def _generate_table_rows(self, experiment_infos: List[ExperimentInfo]) -> List[ModelRow]:
        """Method to generate table rows from remote path to training app save directory"""

        def process_experiment_info(experiment_info: ExperimentInfo):
            try:
                logger.debug(f"Processing experiment info: {experiment_info.task_id}")
                project_info = self._project_infos_map.get(experiment_info.project_id)
                model_row = ExperimentSelector.ModelRow(
                    api=self.api,
                    team_id=self.team_id,
                    task_type=experiment_info.task_type,
                    experiment_info=experiment_info,
                    project_info=project_info,
                )

                def this_row_checkpoint_changed(checkpoint_value: str):
                    self._checkpoint_changed(model_row, checkpoint_value)

                model_row.checkpoint_changed = this_row_checkpoint_changed
                return experiment_info.task_type, model_row
            except Exception as e:
                logger.debug(f"Failed to process experiment info: {experiment_info}")
                return None, None

        table_rows = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_experiment_info, experiment_info)
                for experiment_info in experiment_infos
            ]

            for future in futures:
                result = future.result()
                if result:
                    task_type, model_row = result
                    if task_type is not None and model_row is not None:
                        table_rows.append(model_row)

        table_rows.sort(key=lambda x: x.task_id, reverse=True)
        return table_rows

    def _update_search_text(self):
        self._rows_search_texts = [row.search_text() for row in self._rows]

    def _update_sort_values(self):
        self._rows_sort_values = [row.sort_values() for row in self._rows]

    def _update_value_index_map(self):
        self._first_column_value_to_index = {}
        for i, row in self.table._source_data.iterrows():
            value = row.iloc[0]
            self._first_column_value_to_index[value] = i

    def set_experiment_infos(self, experiment_infos: List[ExperimentInfo]) -> None:
        """
        Updates the experiment infos and regenerates the table rows.
        """
        table_rows = self._generate_table_rows(experiment_infos)
        self._rows = table_rows
        for row in table_rows:
            self.table.insert_row(row.to_table_row())
        self._update_value_index_map()
        self._update_search_text()
        self._update_sort_values()

    def get_selected_experiment_info(self) -> Union[ExperimentInfo, None]:
        selected_row = self.table.get_selected_row()
        if selected_row is None:
            return None
        return self._rows[selected_row.row_index]._experiment_info

    def get_selected_experiment_info_json(self) -> Union[dict, None]:
        experiment_info = self.get_selected_experiment_info()
        if experiment_info is None:
            return None
        return experiment_info.to_json()

    def get_selected_checkpoint_name(self) -> Union[str, None]:
        selected_row = self.table.get_selected_row()
        if selected_row is None:
            return None
        return self._rows[selected_row.row_index].get_selected_checkpoint_name()

    def get_selected_checkpoint_path(self) -> Union[str, None]:
        selected_row = self.table.get_selected_row()
        if selected_row is None:
            return None
        return self._rows[selected_row.row_index].get_selected_checkpoint_path()

    def set_selected_row_by_experiment_info(self, experiment_info: ExperimentInfo) -> None:
        for idx, row in enumerate(self._rows):
            if row._experiment_info.task_id == experiment_info.task_id:
                self.table.select_row(idx)
                return
        raise ValueError(f"Experiment info {experiment_info} not found in the table rows.")

    def _checkpoint_changed(self, row: ModelRow, checkpoint_value: str):
        if self._checkpoint_changed_func is None:
            return
        return self._checkpoint_changed_func(row, checkpoint_value)

    def checkpoint_changed(self, func: Callable[[ModelRow, str], None]):
        self._checkpoint_changed_func = func
        return self._checkpoint_changed_func

    def selection_changed(self, func):
        def f(selected_row: FastTable.ClickedRow):
            if selected_row is None:
                return
            idx = selected_row.row_index
            experiment_info = self._rows[idx]._experiment_info
            func(experiment_info)

        return self.table.selection_changed(f)

    def set_selected_checkpoint_by_name(self, checkpoint_name: str):
        selected_row = self.table.get_selected_row()
        if selected_row is None:
            return
        self._rows[selected_row.row_index].set_selected_checkpoint_by_name(checkpoint_name)

    def set_selected_row_by_task_id(self, task_id: int):
        for idx, row in enumerate(self._rows):
            if row._experiment_info.task_id == task_id:
                self.table.select_row(idx)
                return
        raise ValueError(f"Experiment info with task id {task_id} not found in the table rows.")

    def get_selected_row_by_task_id(self, task_id: int):
        for idx, row in enumerate(self._rows):
            if row._experiment_info.task_id == task_id:
                return row
        return None

    def search(self, search_value: str):
        self.table.search(search_value)

    def disable(self):
        return self.table.disable()

    def enable(self):
        return self.table.enable()

    @property
    def loading(self):
        return self.table.loading

    @loading.setter
    def loading(self, value: bool):
        self.table.loading = value

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def to_html(self):
        return self.table.to_html()
