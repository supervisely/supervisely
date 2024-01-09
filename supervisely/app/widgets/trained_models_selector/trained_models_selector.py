from datetime import datetime
from typing import List, Dict, Union, Literal
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget, ProjectThumbnail, Select, Text, Container, Flexbox
import os
from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.api.file_api import FileInfo
from supervisely.nn.inference.checkpoints.checkpoint import CheckpointInfo

WEIGHTS_DIR = "weights"

COL_ID = "task id".upper()
COL_PROJECT = "training data".upper()
COL_ARTIFACTS = "artifacts".upper()
COL_SESSION = "session".upper()

columns = [
    COL_ID,
    COL_PROJECT,
    COL_ARTIFACTS,
    COL_SESSION,
]


class TrainedModelsSelector(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    class ModelRow:
        def __init__(
            self,
            api: Api,
            team_id: int,
            checkpoint_info: CheckpointInfo,
        ):
            self._api = api
            self._team_id = team_id

            task_id = checkpoint_info.session_id
            task_path = checkpoint_info.session_path
            training_project_name = checkpoint_info.training_project_name
            artifacts = checkpoint_info.artifacts
            session_link = checkpoint_info.session_link

            # col 1 task
            self._task_id = task_id
            self._task_path = task_path
            task_info = self._api.task.get_info_by_id(task_id)
            self._task_date_iso = task_info["startedAt"]
            self._task_date = self._normalize_date()
            self._task_link = self._create_task_link()

            # col 2 project
            self._training_project_name = training_project_name

            # need optimization
            project_info_dummy = self._api.project.get_info_by_name(
                task_info["workspaceId"], self._training_project_name
            )

            if project_info_dummy is not None:
                self._training_project_info = self._api.project.get_info_by_id(
                    project_info_dummy.id
                )
            else:
                self._training_project_info = None

            # col 3 artifacts
            self._artifacts = artifacts
            self._artifacts_names = [artifact_info["name"] for artifact_info in self._artifacts]
            self._artifacts_paths = [artifact_info["path"] for artifact_info in self._artifacts]

            # col 4 session
            self._session_link = session_link

            # widgets
            self._task_widget = self._create_task_widget()
            self._training_project_widget = self._create_training_project_widget()
            self._artifacts_widget = self._create_artifacts_widget()
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
        def training_project_info(self) -> ProjectInfo:
            return self._training_project_info

        @property
        def artifacts_names(self) -> List[str]:
            return self._artifacts_names

        @property
        def artifacts_paths(self) -> List[str]:
            return self._artifacts_paths

        @property
        def artifacts_selector(self) -> Select:
            return self._artifacts_widget

        @property
        def session_link(self) -> str:
            return self._session_link

        def get_selected_artifact_path(self) -> str:
            return self._artifacts_widget.get_value()

        def get_selected_artifact_name(self) -> str:
            return self._artifacts_widget.get_label()

        def to_html(self) -> List[str]:
            return [
                self._task_widget.to_html(),
                self._training_project_widget.to_html(),
                self._artifacts_widget.to_html(),
                self._session_widget.to_html(),
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
                        f"<a href='{self._task_link}'>{self._task_id}</a> <i class='zmdi zmdi-link'></i>",
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

        def _create_artifacts_widget(self) -> Select:
            artifact_selector = Select(
                [
                    Select.Item(value=artifact_info["path"], label=artifact_info["name"])
                    for artifact_info in self._artifacts
                ]
            )
            return artifact_selector

        def _create_session_widget(self) -> Text:
            session_link_widget = Text(
                f"<a href='{self._session_link}'>Preview</a> <i class='zmdi zmdi-link'></i>", "text"
            )
            return session_link_widget

    def __init__(
        self,
        team_id: int,
        checkpoint_infos: List[CheckpointInfo],
        widget_id: str = None,
    ):
        self._api = Api.from_env()

        self._team_id = team_id
        table_rows = self._generate_table_rows(checkpoint_infos)

        self._columns = columns
        self._rows = table_rows
        self._rows_html = [row.to_html() for row in self._rows]
        self._changes_handled = False

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
            "rows_html": self._rows_html,
        }

    def get_json_state(self) -> Dict:
        return {"selectedRow": 0}

    def _generate_table_rows(self, checkpoint_infos: List[CheckpointInfo]) -> List[Dict]:
        """Method to generate table rows from remote path to training app save directory"""
        table_rows = []
        for checkpoint_info in checkpoint_infos:
            table_rows.append(
                TrainedModelsSelector.ModelRow(
                    api=self._api,
                    team_id=self._team_id,
                    checkpoint_info=checkpoint_info,
                )
            )
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

    def select_row(self, row_index):
        if row_index < 0 or row_index > len(self._rows) - 1:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()

    def value_changed(self, func):
        route_path = self.get_route_path(TrainedModelsSelector.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True
        print(self._changes_handled)

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_row()
            func(res)

        return _value_changed
