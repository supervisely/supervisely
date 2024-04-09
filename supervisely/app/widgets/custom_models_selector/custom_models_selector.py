import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union, Callable

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.api.file_api import FileApi, FileInfo
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Container,
    Field,
    Checkbox,
    Flexbox,
    ProjectThumbnail,
    Select,
    Text,
    Input,
    Button,
    FileThumbnail,
    Widget,
)
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo
from supervisely import env
from supervisely.io.fs import get_file_name_with_ext

WEIGHTS_DIR = "weights"

COL_ID = "task id".upper()
COL_PROJECT = "training data".upper()
COL_CHECKPOINTS = "checkpoints".upper()
COL_SESSION = "session".upper()

columns = [
    COL_ID,
    COL_PROJECT,
    COL_CHECKPOINTS,
    COL_SESSION,
]


class CustomModelsSelector(Widget):
    class Routes:
        TASK_TYPE_CHANGED = "task_type_changed"
        VALUE_CHANGED = "value_changed"

    class ModelRow:
        def __init__(
            self,
            api: Api,
            team_id: int,
            checkpoint: CheckpointInfo,
            task_type: str,
        ):
            self._api = api
            self._team_id = team_id
            self._task_type = task_type

            task_id = checkpoint.session_id
            if type(task_id) is str:
                if task_id.isdigit():
                    task_id = int(task_id)
                else:
                    raise ValueError(f"Task id {task_id} is not a number")

            task_path = checkpoint.session_path
            training_project_name = checkpoint.training_project_name
            checkpoints = checkpoint.checkpoints
            session_link = checkpoint.session_link

            # col 1 task
            self._task_id = task_id
            self._task_path = task_path
            task_info = self._api.task.get_info_by_id(task_id)
            self._task_date_iso = task_info["startedAt"]
            self._task_date = self._normalize_date()
            self._task_link = self._create_task_link()

            # col 2 project
            self._training_project_name = training_project_name

            project_info = self._api.project.get_info_by_name(
                task_info["workspaceId"], self._training_project_name
            )
            self._training_project_info = project_info

            # col 3 checkpoints
            self._checkpoints = checkpoints
            self._checkpoints_names = [
                checkpoint_info["name"] for checkpoint_info in self._checkpoints
            ]
            self._checkpoints_paths = [
                checkpoint_info["path"] for checkpoint_info in self._checkpoints
            ]

            # col 4 session
            self._session_link = session_link

            # widgets
            self._task_widget = self._create_task_widget()
            self._training_project_widget = self._create_training_project_widget()
            self._checkpoints_widget = self._create_checkpoints_widget()
            self._session_widget = self._create_session_widget()

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

        def get_selected_checkpoint_path(self) -> str:
            return self._checkpoints_widget.get_value()

        def get_selected_checkpoint_name(self) -> str:
            return self._checkpoints_widget.get_label()

        def to_html(self) -> List[str]:
            return [
                f"<div> {self._task_widget.to_html()} </div>",
                f"<div> {self._training_project_widget.to_html()} </div>",
                f"<div> {self._checkpoints_widget.to_html()} </div>",
                f"<div> {self._session_widget.to_html()} </div>",
            ]

        def _normalize_date(self) -> str:
            date_obj = datetime.fromisoformat(self._task_date_iso.rstrip("Z"))
            formatted_date = date_obj.strftime("%d %B %Y, %H:%M")
            return formatted_date

        def _create_task_link(self) -> str:
            remote_path = os.path.join(self._task_path, "open_app.lnk")
            task_file = self._api.file.get_info_by_path(self._team_id, remote_path)
            if task_file is not None:
                return f"{self._api.server_address}/files/{task_file.id}"
            else:
                return ""

        def _create_task_widget(self) -> Flexbox:
            task_widget = Container(
                [
                    Text(
                        f"<i class='zmdi zmdi-folder' style='color: #7f858e'></i> <a href='{self._task_link}'>{self._task_id}</a>",
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
            checkpoint_selector = Select(
                [
                    Select.Item(value=checkpoint_info["path"], label=checkpoint_info["name"])
                    for checkpoint_info in self._checkpoints
                ]
            )
            return checkpoint_selector

        def _create_session_widget(self) -> Text:
            session_link_widget = Text(
                f"<a href='{self._session_link}'>Preview</a> <i class='zmdi zmdi-open-in-new'></i>",
                "text",
            )
            return session_link_widget

    def __init__(
        self,
        team_id: int,
        checkpoints: List[CheckpointInfo],
        show_custom_checkpoint_path: bool = False,
        custom_checkpoint_task_types: List[str] = [],
        widget_id: str = None,
    ):
        self._api = Api.from_env()

        self._team_id = team_id
        table_rows = self._generate_table_rows(checkpoints)
        self._show_custom_checkpoint_path = show_custom_checkpoint_path
        self._custom_checkpoint_task_types = custom_checkpoint_task_types

        self._columns = columns
        self._rows = table_rows
        # self._rows_html = #[row.to_html() for row in self._rows]

        task_types = []
        self._rows_html = defaultdict(list)
        for model_row in table_rows:
            task_types.append(model_row.task_type)
            self._rows_html[model_row.task_type].append(model_row.to_html())

        self._task_types = list(set(task_types))
        if len(self._task_types) == 0:
            self.__default_selected_task_type = None
        else:
            self.__default_selected_task_type = self._task_types[0]

        self._changes_handled = False
        self._task_changes_handled = False

        if self._show_custom_checkpoint_path:
            self.file_thumbnail = FileThumbnail()
            team_files_url = f"{env.server_address().rstrip('/')}/files/"

            team_files_link_btn = Button(
                text="Open Team Files",
                button_type="info",
                plain=True,
                icon="zmdi zmdi-folder",
                link=team_files_url,
            )

            file_api = FileApi(self._api)
            self._model_path_input = Input(placeholder="Path to model file in Team Files")

            @self._model_path_input.value_changed
            def change_folder(value):
                file_info = None
                if value != "":
                    file_info = file_api.get_info_by_path(env.team_id(), value)
                self.file_thumbnail.set(file_info)

            model_path_field = Field(
                self._model_path_input,
                title=f"Copy path to model file from Team Files and paste to field below.",
                description="Copy path in Team Files",
            )
            self.custom_checkpoint_task_type_selector_field = None
            if len(self._custom_checkpoint_task_types) > 0:
                self.custom_checkpoint_task_type_selector_items = [
                    Select.Item(value=task_type, label=task_type)
                    for task_type in self._custom_checkpoint_task_types
                ]
                self.custom_checkpoint_task_type_selector = Select(
                    self.custom_checkpoint_task_type_selector_items
                )
                custom_checkpoint_task_type_selector_field = Field(
                    title="Task Type", content=self.custom_checkpoint_task_type_selector
                )

            self.custom_tab_widgets = Container(
                [
                    team_files_link_btn,
                    custom_checkpoint_task_type_selector_field,
                    model_path_field,
                    self.file_thumbnail,
                ]
            )

            self.custom_tab_widgets.hide()

            self.show_custom_checkpoint_path_checkbox = Checkbox(
                "Use custom path to checkpoint", False
            )

            @self.show_custom_checkpoint_path_checkbox.value_changed
            def show_custom_checkpoint_path_checkbox_changed(is_checked):
                if is_checked:
                    self.disable_table()
                    self.custom_tab_widgets.show()
                else:
                    self.enable_table()
                    self.custom_tab_widgets.hide()

                StateJson()[self.widget_id]["useCustomPath"] = is_checked
                StateJson().send_changes()

        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def rows(self) -> List[ModelRow]:
        return self._rows

    def get_json_data(self) -> Dict:
        return {
            "columns": self._columns,
            "rowsHtml": self._rows_html,
            "taskTypes": self._task_types,
        }

    def get_json_state(self) -> Dict:
        if self._show_custom_checkpoint_path:
            return {
                "selectedRow": 0,
                "selectedTaskType": self.__default_selected_task_type,
                "useCustomPath": False,
            }
        else:
            return {"selectedRow": 0}

    def set_active_task_type(self, task_type: str):
        if task_type not in self._task_types:
            raise ValueError(f'Task type "{task_type}" does not exist')
        StateJson()[self.widget_id]["selectedTaskType"] = task_type
        StateJson().send_changes()

    def get_available_task_types(self) -> List[str]:
        return self._task_types

    def disable_table(self) -> None:
        for row in self._rows:
            row.checkpoints_selector.disable()
        super().disable()

    def enable_table(self) -> None:
        for row in self._rows:
            row.checkpoints_selector.enable()
        super().enable()

    def enable(self):
        self.custom_tab_widgets.enable()
        self._model_path_input.enable()
        self.custom_checkpoint_task_type_selector.enable()
        self.show_custom_checkpoint_path_checkbox.enable()
        self.enable_table()
        super().enable()

    def disable(self) -> None:
        self.custom_tab_widgets.disable()
        self._model_path_input.disable()
        self.custom_checkpoint_task_type_selector.disable()
        self.show_custom_checkpoint_path_checkbox.disable()
        self.disable_table()
        super().disable()

    def _generate_table_rows(self, checkpoint_infos: List[CheckpointInfo]) -> List[Dict]:
        """Method to generate table rows from remote path to training app save directory"""
        table_rows = []
        for checkpoint_info in checkpoint_infos:
            try:
                model_row = CustomModelsSelector.ModelRow(
                    api=self._api,
                    team_id=self._team_id,
                    checkpoint=checkpoint_info,
                    task_type=checkpoint_info.task_type,
                )
                table_rows.append(model_row)
            except:
                continue
        return table_rows

    def get_selected_row(self, state=StateJson()) -> Union[ModelRow, None]:
        if len(self._rows) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = int(widget_actual_state["selectedRow"])
            return self._rows[selected_row_index]

    def get_selected_row_index(self, state=StateJson()) -> Union[int, None]:
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

    def get_selected_task_type(self) -> str:
        return StateJson()[self.widget_id]["selectedTaskType"]

    def get_selected_model_params(self) -> Union[Dict, None]:
        is_custom_path = self.use_custom_checkpoint_path()
        if not is_custom_path:
            selected_model = self.get_selected_row()
            task_type = selected_model.task_type
            checkpoint_filename = selected_model.get_selected_checkpoint_name()
            checkpoint_url = selected_model.get_selected_checkpoint_path()
        else:
            task_type = self.get_custom_checkpoint_task_type()
            checkpoint_filename = self.get_custom_checkpoint_name()
            checkpoint_url = self.get_custom_checkpoint_path()

        model_params = {
            "model_source": "Custom models",
            "task_type": task_type,
            "checkpoint_name": checkpoint_filename,
            "checkpoint_url": checkpoint_url,
        }
        return model_params

    def set_active_row(self, row_index: int) -> None:
        if row_index < 0 or row_index > len(self._rows) - 1:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()

    def use_custom_checkpoint_path(self) -> bool:
        return self.show_custom_checkpoint_path_checkbox.is_checked()

    def get_custom_checkpoint_name(self) -> str:
        if self.use_custom_checkpoint_path():
            return get_file_name_with_ext(self.get_custom_checkpoint_path())

    def get_custom_checkpoint_path(self) -> str:
        if self.use_custom_checkpoint_path():
            return self._model_path_input.get_value()

    def set_custom_checkpoint_path(self, path: str) -> None:
        if self.use_custom_checkpoint_path():
            self._model_path_input.set_value(path)

    def set_custom_checkpoint_preview(self, file_info: FileInfo) -> None:
        if self.use_custom_checkpoint_path():
            self.file_thumbnail.set(file_info)

    def get_custom_checkpoint_task_type(self) -> str:
        if self.use_custom_checkpoint_path():
            return self.custom_checkpoint_task_type_selector.get_value()

    def set_custom_checkpoint_task_type(self, task_type: str) -> None:
        if self.use_custom_checkpoint_path():
            available_task_types = self.custom_checkpoint_task_type_selector.get_labels()
            if task_type not in available_task_types:
                raise ValueError(f'"{task_type}" is not available task type')
            self.custom_checkpoint_task_type_selector.set_value(task_type)

    def task_type_changed(self, func: Callable):
        route_path = self.get_route_path(CustomModelsSelector.Routes.TASK_TYPE_CHANGED)
        server = self._sly_app.get_server()
        self._task_changes_handled = True

        @server.post(route_path)
        def _task_type_changed():
            res = self.get_selected_task_type()
            func(res)

        return _task_type_changed

    def value_changed(self, func: Callable):
        route_path = self.get_route_path(CustomModelsSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_row()
            func(res)

        return _value_changed
