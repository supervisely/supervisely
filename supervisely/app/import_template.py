from typing import Optional, Union, List
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, basename
from pathlib import Path
import supervisely as sly

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal

from supervisely.app.widgets import (
    Container,
    Card,
    Checkbox,
    Text,
    FileStorageUpload,
    Button,
    DestinationProject,
    RadioTabs,
    # TeamFilesSelector,
)
from supervisely.project.project_type import ProjectType


class Import:
    def __init__(self) -> None:
        self._team_id = env.team_id()
        self._workspace_id = env.workspace_id()
        self._file = env.file(raise_not_found=False)
        self._folder = env.folder(raise_not_found=False)
        self._project_id = env.project_id(raise_not_found=False)
        self._dataset_id = env.dataset_id(raise_not_found=False)
        self._app = None
        self._run_button = None

    def initiate_import_widgets(
        self,
        input_path: str = "/import",
        current_widgets: List = [],
        project_type: ProjectType = ProjectType.IMAGES,
    ):

        if self._file is not None or self._folder is not None:
            if self._file is not None:
                current_path = self._file
            else:
                current_path = self._folder
            import_path_text = Text(text=f"Data will be import from {current_path}", status="info")
            input_card = Card(
                title="Input Menu",
                content=Container(widgets=[import_path_text]),
            )
        else:
            self.file_upload = FileStorageUpload(
                team_id=self._team_id, path=input_path, change_name_if_conflict=True
            )
            self.team_files = Text(text="Team Files here")  # TODO check it
            self.drag_drop_title = "Drag & drop"
            self.team_files_title = "Team files"
            titles = [self.drag_drop_title, self.team_files_title]
            contents = [self.file_upload, self.team_files]
            self.radio_tabs = RadioTabs(titles=titles, contents=contents)
            widgets = Container(widgets=[self.radio_tabs])
            card_upload_or_tf = Card(content=widgets)
            input_card = Card(
                title="Input Menu",
                content=Container(widgets=[card_upload_or_tf]),
            )

        temporary_files_main_text = Text(text="Remove temporary files after successful import")
        temporary_files_add_text = Text(
            text="Removes source directory from Team Files after successful import"
        )
        temporary_files_checkboxes_data = Container(
            widgets=[temporary_files_main_text, temporary_files_add_text], direction="vertical"
        )
        self.temporary_files = Checkbox(content=temporary_files_checkboxes_data, checked=True)

        checkbox_data = Container(
            widgets=[
                self.temporary_files,
            ],
            direction="vertical",
            gap=15,
        )

        checkbox = Card(
            content=checkbox_data,
        )

        if self._project_id is not None or self._dataset_id is not None:
            if self._project_id is not None:
                import_project_text = Text(
                    text=f"Data will be import in project with ID {self._project_id}", status="info"
                )
            else:
                import_project_text = Text(
                    text=f"Data will be import in dataset with ID {self._dataset_id}", status="info"
                )
            destination_card = Card(content=import_project_text)
        else:
            self.destination_project = DestinationProject(
                workspace_id=self._workspace_id, project_type=project_type
            )

            destination_card = Card(
                content=self.destination_project,
            )

        self._run_button = Button(text="Run import")

        if len(current_widgets) > 0:
            current_app_container = Container(widgets=current_widgets)
            self.current_app_widgets = Card(
                title="Current APP widgets", content=current_app_container
            )

            layout = Container(
                widgets=[
                    input_card,
                    checkbox,
                    self.current_app_widgets,
                    destination_card,
                    self.run_button,
                ],
                direction="vertical",
                gap=15,
            )

        else:
            layout = Container(
                widgets=[
                    input_card,
                    checkbox,
                    destination_card,
                    self.run_button,
                ],
                direction="vertical",
                gap=15,
            )

        self._app = sly.Application(layout=layout)

    @property
    def app(self):
        return self._app

    @property
    def run_button(self):
        return self._run_button

    def process(self):
        raise NotImplementedError()  # implement your own method when inherit

    # project_id = self.process()
    # if type(project_id) is int and is_production():
    #     info = api.project.get_info_by_id(project_id)
    #     api.task.set_output_project(task_id=task_id, project_id=info.id, project_name=info.name)
    #     print(f"Result project: id={info.id}, name={info.name}")
