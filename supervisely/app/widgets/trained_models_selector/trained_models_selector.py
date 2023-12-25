from datetime import datetime
from typing import List, Dict, Union, Literal
from supervisely.app.jinja2 import create_env
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import Widget, ProjectThumbnail, Select, Text, Container, Flexbox
import os
from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.api.file_api import FileInfo

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
            task_id: str,
            task_path: str,
            training_project_name: str,
            artifacts_infos: List[FileInfo],
            session_link: str,
        ):
            self._api = api
            self._team_id = team_id

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
            self._training_project_info = self._api.project.get_info_by_id(project_info_dummy.id)

            # col 3 artifacts
            self._artifacts_infos = artifacts_infos
            self._artifacts_names = [artifact_info.name for artifact_info in self._artifacts_infos]
            self._artifacts_paths = [artifact_info.path for artifact_info in self._artifacts_infos]

            # col 4 session
            self._session_link = session_link

            # widgets
            self._task_widget = self._create_task_widget()
            self._training_project_widget = self._create_training_project_widget()
            self._artifacts_widget = self._create_artifacts_widget()
            self._session_widget = self._create_session_widget()

        @property
        def task_id(self):
            return self._task_id

        @property
        def task_date(self):
            return self._task_date

        @property
        def task_link(self):
            return self._task_link

        @property
        def training_project_info(self):
            return self._training_project_info

        @property
        def get_artifacts_names(self):
            return self._artifacts_names

        @property
        def artifacts_selector(self):
            return self._artifacts_widget

        @property
        def session_link(self):
            return self._session_link

        def to_html(self):
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
            task_file_id = self._api.file.get_info_by_path(self._team_id, remote_path).id
            return f"{self._api.server_address}/files/{task_file_id}"

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

        def _create_training_project_widget(self) -> ProjectThumbnail:
            training_project_widget = ProjectThumbnail(
                self._training_project_info, remove_margins=True
            )
            return training_project_widget

        def _create_artifacts_widget(self) -> Select:
            artifact_selector = Select(
                [
                    Select.Item(value=artifact_info.path, label=artifact_info.name)
                    for artifact_info in self._artifacts_infos
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
        remote_path_to_models: str,
        task_type: Literal["instance segmentation", "object detection", "pose estimation"],
        widget_id: str = None,
    ):
        self._api = Api.from_env()

        self._team_id = team_id
        self._remote_path_to_models = remote_path_to_models
        self._task_type = task_type
        table_rows = self._generate_table_rows()

        self._columns = columns
        self._rows = table_rows
        self._rows_html = [row.to_html() for row in self._rows]
        self._changes_handled = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "columns": self._columns,
            "rows_html": self._rows_html,
        }

    def get_json_state(self):
        return {"selectedRow": 0}

    def _generate_table_rows(self) -> List[Dict]:
        """Method to generate table rows from remote path to models"""
        table_rows = []
        path_to_projects = os.path.join(self._remote_path_to_models, f"{self._task_type}")
        project_files_infos = self._api.file.list(
            self._team_id, path_to_projects, recursive=False, return_type="fileinfo"
        )
        for project_file_info in project_files_infos:
            project_name = project_file_info.name
            task_files_infos = self._api.file.list(
                self._team_id, project_file_info.path, recursive=False, return_type="fileinfo"
            )
            for task_file_info in task_files_infos:
                if task_file_info.name == "images":
                    continue
                task_id = task_file_info.name
                session_link = f"{self._api.server_address}/apps/sessions/{task_id}"
                paths_to_artifacts = os.path.join(task_file_info.path, WEIGHTS_DIR)
                artifacts_infos = self._api.file.list(
                    self._team_id, paths_to_artifacts, recursive=False, return_type="fileinfo"
                )

                table_rows.append(
                    TrainedModelsSelector.ModelRow(
                        api=self._api,
                        team_id=self._team_id,
                        task_id=task_id,
                        task_path=task_file_info.path,
                        training_project_name=project_name,
                        artifacts_infos=artifacts_infos,
                        session_link=session_link,
                    )
                )
        return table_rows

    def get_selected_row(self, state=StateJson()):
        if len(self._rows) == 0:
            return
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            selected_row_index = int(widget_actual_state["selectedRow"])
            return self._rows[selected_row_index]

    def get_selected_row_index(self, state=StateJson()):
        widget_actual_state = state[self.widget_id]
        widget_actual_data = DataJson()[self.widget_id]
        if widget_actual_state is not None and widget_actual_data is not None:
            return widget_actual_state["selectedRow"]

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

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value
        DataJson()[self.widget_id]["rows"] = self._rows
        DataJson().send_changes()

    def select_row(self, row_index):
        if row_index < 0 or row_index > len(self._rows) - 1:
            raise ValueError(f'Row with index "{row_index}" does not exist')
        StateJson()[self.widget_id]["selectedRow"] = row_index
        StateJson().send_changes()
