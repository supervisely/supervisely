from typing import Optional, Union
from supervisely._utils import is_production, is_development
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, basename
from supervisely.sly_logger import logger
from supervisely.io.fs import dir_exists, file_exists, remove_dir, silent_remove
from supervisely.app.fastapi.subapp import Application

import supervisely as sly
from supervisely.app.widgets import (
    Card,
    FileStorageUpload,
    SelectProject,
    SelectDataset,
    SelectWorkspace,
    TeamFilesSelector,
    Container,
    Tabs,
    Button,
    Input,
    Empty,
    Text,
    ProjectThumbnail,
    DatasetThumbnail,
    FileThumbnail,
    FolderThumbnail,
    SlyTqdm,
    Widget,
)

# @TODO: set project type in constructor?
# @TODO: if run from team files hide input card
# @TODO: if run from project/dataset hide output card
# @TODO: build with GUI
# @TODO: check agent import
# @TODO: remove source checkbox working for UI and non UI imports
# @TODO: scenarios for import from different sources:
# @TODO: 1. import from scractch (without template)
# @TODO: 2. import with template from file .txt (links)
# @TODO: 3. import with template from file archive
# @TODO: 4. import with template from folder
# @TODO: 6. import with template from external link
# @TODO: 5. import with template from agent folder
# @TODO: 7. import with template GUI


class Import(Application):
    class Context:
        def __init__(
            self,
            team_id: int,
            workspace_id: int,
            project_id: int,
            dataset_id: int,
            path: str,
            is_directory: bool = True,
            is_on_agent: bool = False,
            is_path_required: bool = True,
        ):
            self._team_id = team_id
            if self._team_id is None:
                raise ValueError(f"Team ID is not specified: {self._team_id}")
            if type(self._team_id) is not int:
                raise ValueError(f"Team ID must be 'int': {self._team_id}")

            self._workspace_id = workspace_id
            if self._workspace_id is None:
                raise ValueError(f"Workspace ID is not specified: {self._workspace_id}")
            if type(self._workspace_id) is not int:
                raise ValueError(f"Workspace ID must be 'int': {self._workspace_id}")

            self._project_id = project_id
            if self._project_id is not None and type(self._project_id) is not int:
                raise ValueError(f"Project ID must be 'int': {self._project_id}")

            self._dataset_id = dataset_id
            if self._dataset_id is not None and type(self._dataset_id) is not int:
                raise ValueError(f"Dataset ID must be 'int': {self._dataset_id}")

            self._is_path_required = is_path_required
            if type(self._is_path_required) is not bool:
                raise ValueError(
                    f"Flag 'is_path_required' must be 'bool': {self._is_path_required}"
                )

            self._path = path
            # if self._is_path_required is True and self._path is None:
            #     raise ValueError(f"Remote path is not specified: {self._path}")
            # if self._is_path_required is True and type(self._path) is not str:
            #     raise ValueError(f"Remote path must be 'str': {self._path}")

            self._is_directory = is_directory
            if self._is_path_required is True and type(self._is_directory) is not bool:
                raise ValueError(f"Flag 'is_directory' must be 'bool': {self._is_directory}")

            self._is_on_agent = is_on_agent
            if self._is_path_required is True and type(self._is_on_agent) is not bool:
                raise ValueError(f"Flag 'is_on_agent' must be 'bool': {self._is_on_agent}")

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
                f"Path: {self._path}\n"
                f"Is directory: {self._is_directory}\n"
                f"Is on agent: {self._is_on_agent}"
            )

        @property
        def team_id(self) -> int:
            return self._team_id

        @property
        def workspace_id(self) -> int:
            return self._workspace_id

        @property
        def project_id(self) -> int:
            return self._project_id

        @property
        def dataset_id(self) -> int:
            return self._dataset_id

        @property
        def path(self) -> str:
            return self._path

        @property
        def is_directory(self) -> bool:
            return self._is_directory

        @property
        def is_on_agent(self) -> bool:
            return self._is_on_agent

    def __init__(self):
        self._layout = None
        if sly.is_production():
            api = Api.from_env()

            ####################
            # PROJECT SELECTOR #
            ####################

            self.input_mode = None

            # Launch from Dataset
            if (
                sly.env.project_id(raise_not_found=False) is not None
                and sly.env.dataset_id(raise_not_found=False) is not None
            ):
                project_info = api.project.get_info_by_id(sly.env.project_id())
                dataset_info = api.dataset.get_info_by_id(sly.env.dataset_id())
                self.input_thumbnail = DatasetThumbnail(
                    project_info=project_info, dataset_info=dataset_info
                )
                self.project_selector_container = Container(widgets=[self.input_thumbnail])

                self.input_mode = "dataset"

            # Launch from Project
            elif sly.env.project_id(raise_not_found=False) is not None:
                project_info = api.project.get_info_by_id(sly.env.project_id())
                self.input_thumbnail = ProjectThumbnail(info=project_info)
                self.project_selector_container = Container(widgets=[self.input_thumbnail])
                self.input_mode = "project"

            # Launch from Ecosystem
            else:
                self.output_workspace_selector = SelectWorkspace()
                self.output_new_project_name = Input(value="My project")
                self.output_new_project = Container(
                    widgets=[
                        self.output_workspace_selector,
                        Text("Project name"),
                        self.output_new_project_name,
                    ]
                )
                self.output_project_selector = SelectProject()
                self.output_dataset_selector = SelectDataset()
                self.output_project_tabs = Tabs(
                    labels=["New Project", "Existing Project", "Existing Dataset"],
                    contents=[
                        self.output_new_project,
                        self.output_project_selector,
                        self.output_dataset_selector,
                    ],
                )
                self.project_selector_container = Container(widgets=[self.output_project_tabs])
                self.input_mode = "ecosystem"

            ####################
            #  FILES SELECTOR  #
            ####################

            # Launch from File
            if sly.env.file(raise_not_found=False) is not None:
                file_info = api.file.get_info_by_path(
                    team_id=sly.env.team_id(), remote_path=sly.env.file()
                )
                self.input_thumbnail = FileThumbnail(info=file_info)
                self.files_selector_container = Container(widgets=[self.input_thumbnail])
                if self.input_mode is None or self.input_mode == "ecosystem":
                    self.input_mode = "file"

            # Launch from Folder
            elif sly.env.folder(raise_not_found=False) is not None:
                file_info = api.file.get_info_by_path(
                    team_id=sly.env.team_id(), remote_path=sly.env.folder()
                )
                self.input_thumbnail = FolderThumbnail(info=file_info)
                self.files_selector_container = Container(widgets=[self.input_thumbnail])
                if self.input_mode is None or self.input_mode == "ecosystem":
                    self.input_mode = "folder"

            # Launch from Ecosystem
            else:
                self.input_file_selector = TeamFilesSelector(
                    team_id=sly.env.team_id(), multiple_selection=False, max_height=300
                )

                app_name = sly.env.app_name(raise_not_found=False)
                if app_name is None:
                    app_name = "import-template"
                else:
                    app_name = app_name.lower().replace(" ", "-")
                self.storage_upload_path = f"/import/{app_name}/"
                self.input_drag_n_drop = FileStorageUpload(
                    team_id=sly.env.team_id(), path=self.storage_upload_path
                )
                self.input_external_link = Input(value="https://")
                self.input_data_tabs = Tabs(
                    labels=["File Selector", "Drag & Drop", "Link"],
                    contents=[
                        self.input_file_selector,
                        self.input_drag_n_drop,
                        self.input_external_link,
                    ],
                )
                self.files_selector_container = Container(widgets=[self.input_data_tabs])
                if self.input_mode is None:
                    self.input_mode = "ecosystem"

            ############
            # SETTINGS #
            #############

            self.settings_card = self.generate_settings_card()

            ##########
            # LAYOUT #
            ##########

            if self.input_mode == "project":
                self.input_project_card = Card(
                    title="Input Project",
                    description="Application was launched from the context menu of the project",
                    content=self.project_selector_container,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import from 3 options below. "
                        "You can select files from Team Files, "
                        "use Drag & Drop "
                        "or provide external link for downloading"
                    ),
                    content=self.files_selector_container,
                )
                self._layout = Container(
                    widgets=[
                        self.input_project_card,
                        self.input_files_card,
                    ]
                )

            elif self.input_mode == "dataset":
                self.input_project_card = Card(
                    title="Input Dataset",
                    description="Application was launched from the context menu of the dataset",
                    content=self.project_selector_container,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import from 3 options below. "
                        "You can select files from Team Files, "
                        "use Drag & Drop "
                        "or provide external link for downloading"
                    ),
                    content=self.files_selector_container,
                )
                self._layout = Container(
                    widgets=[
                        self.input_project_card,
                        self.input_files_card,
                    ]
                )

            elif self.input_mode == "file":
                self.input_project_card = Card(
                    title="Select Project",
                    description="Select where do you want to import your data from 3 options below",
                    content=self.project_selector_container,
                )
                self.input_files_card = Card(
                    title="Input File",
                    description="Application was launched from the context menu of the file",
                    content=self.files_selector_container,
                )
                self._layout = Container(
                    widgets=[
                        self.input_files_card,
                        self.input_project_card,
                    ]
                )

            elif self.input_mode == "folder":
                self.input_project_card = Card(
                    title="Select Project",
                    description="Select where do you want to import your data from 3 options below",
                    content=self.project_selector_container,
                )
                self.input_files_card = Card(
                    title="Input File",
                    description="Application was launched from the context menu of the file",
                    content=self.files_selector_container,
                )
                self._layout = Container(
                    widgets=[
                        self.input_files_card,
                        self.input_project_card,
                    ]
                )
            else:
                self.input_project_card = Card(
                    title="Select Project",
                    description="Select where do you want to import your data from 3 options below",
                    content=self.project_selector_container,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import from 3 options below. "
                        "You can select files from Team Files, "
                        "use Drag & Drop "
                        "or provide external link for downloading"
                    ),
                    content=self.files_selector_container,
                )
                self._layout = Container(
                    widgets=[
                        self.input_project_card,
                        self.input_files_card,
                    ]
                )

        self._start_button = Button("Start Import")
        self.import_progress = SlyTqdm()
        self._layout._widgets.extend([self.settings_card, self._start_button, self.import_progress])
        super().__init__(layout=self._layout)

    def generate_settings_card(self) -> Widget:
        # implement your own method for import settings
        return Empty()

    def process(self, context: Context) -> Optional[Union[int, None]]:
        # implement your own method for import
        raise NotImplementedError()

    def is_path_required(self) -> bool:
        return True

    def run(self):
        api = Api.from_env()
        task_id = None
        if is_production():
            task_id = env.task_id()

        team_id = env.team_id()
        workspace_id = env.workspace_id()

        file = env.file(raise_not_found=False)
        folder = env.folder(raise_not_found=False)

        if file is not None and folder is not None:
            raise KeyError(
                "Both FILE and FOLDER envs are defined, but only one is allowed for the import"
            )
        if self.is_path_required() is True:
            if is_development() is True:
                if file is None and folder is None:
                    raise KeyError(
                        "One of the environment variables has to be defined for the import app: FILE or FOLDER"
                    )

        is_directory = True
        path = folder
        if file is not None:
            path = file
            is_directory = False

        remote_path = path

        is_on_agent = False
        if path is not None:
            is_on_agent = api.file.is_on_agent(path)

        if self.is_path_required() is False and path is None:
            is_directory = False

        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        if project_id is not None:
            # lets validate that project exists
            project = api.project.get_info_by_id(id=project_id)
            if project is None:
                raise ValueError(f"Project with ID: '{project_id}' is not found or either archived")
            logger.info(f"Importing to existing Project: id={project.id}, name={project.name}")
        if dataset_id is not None:
            # lets validate that dataset exists
            dataset = api.dataset.get_info_by_id(id=dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset with ID: '{dataset_id}' is not found or either archived")
            logger.info(f"Importing to existing Dataset: id={dataset.id}, name={dataset.name}")

        if is_production():
            if path is not None:
                local_save_path = join(get_data_dir(), basename(path.rstrip("/")))
                if dir_exists(local_save_path):
                    remove_dir(local_save_path)
                if is_directory:
                    api.file.download_directory(
                        team_id=team_id, remote_path=path, local_save_path=local_save_path
                    )
                else:
                    if file_exists(local_save_path):
                        silent_remove(local_save_path)
                    api.file.download(
                        team_id=team_id, remote_path=path, local_save_path=local_save_path
                    )
                path = local_save_path

        if is_development():
            context = Import.Context(
                team_id=team_id,
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=path,
                is_directory=is_directory,
                is_on_agent=is_on_agent,
                is_path_required=self.is_path_required(),
            )
            project_id = self.process(context=context)
            info = api.project.get_info_by_id(project_id)
            logger.info(f"Result project: id={info.id}, name={info.name}")

        if is_production() is True:

            @self._start_button.click
            def start_import():
                self.import_progress.show()
                context = self.__get_context(api=api)
                project_id = self.process(context=context)
                if type(project_id) is int:
                    info = api.project.get_info_by_id(project_id)
                    api.task.set_output_project(
                        task_id=task_id, project_id=info.id, project_name=info.name
                    )
                    logger.info(f"Result project: id={info.id}, name={info.name}")
                self._start_button.disabled = True

    def __get_context(self, api: Api) -> Context:
        team_id = sly.env.team_id()
        workspace_id = sly.env.workspace_id()
        project_id = None
        dataset_id = None
        path = None
        is_directory = True
        is_on_agent = False

        if self.input_mode == "project":
            project_id = sly.env.project_id()
        elif self.input_mode == "dataset":
            project_id = sly.env.project_id()
            dataset_id = sly.env.dataset_id()
        elif self.input_mode == "file":
            is_directory = False
            path = sly.env.file()
        elif self.input_mode == "folder":
            is_directory = True
            path = sly.env.folder()
        else:
            project_mode = self.output_project_tabs.get_active_tab()
            if project_mode == "New Project":
                team_id = self.output_workspace_selector._team_selector.get_selected_id()
                workspace_id = self.output_workspace_selector.get_selected_id()
                project_name = self.output_new_project_name.get_value()
            elif project_mode == "Existing Project":
                team_id = self.output_project_selector._ws_selector._team_selector.get_selected_id()
                workspace_id = self.output_project_selector._ws_selector.get_selected_id()
                project_id = self.output_project_selector.get_selected_id()
                dataset_id = None
            elif project_mode == "Existing Dataset":
                team_id = (
                    self.output_dataset_selector._project_selector._ws_selector._team_selector.get_selected_id()
                )
                workspace_id = (
                    self.output_dataset_selector._project_selector._ws_selector.get_selected_id()
                )
                project_id = self.output_dataset_selector._project_selector.get_selected_id()
                dataset_id = self.output_dataset_selector.get_selected_id()

            data_mode = self.input_data_tabs.get_active_tab()
            data_dir = get_data_dir()
            if data_mode == "File Selector":
                paths = self.input_file_selector.get_selected_paths()

                data_path = self.input_file_selector.get_selected_paths()[0]
                data_type = self.input_file_selector.get_selected_items()[0]["type"]

                path = join(data_dir, basename(data_path.rstrip("/")))
                if data_type == "file":
                    api.file.download(team_id=team_id, remote_path=data_path, local_save_path=path)
                    is_directory = False
                else:
                    api.file.download_directory(
                        team_id=team_id, remote_path=data_path, local_save_path=path
                    )
                    is_directory = True
            elif data_mode == "Drag & Drop":
                paths = self.input_drag_n_drop.get_uploaded_paths()
                for path in paths:
                    local_save_path = join(data_dir, path.replace(self.storage_upload_path, ""))
                    api.file.download(
                        team_id=team_id, remote_path=path, local_save_path=local_save_path
                    )
                path = data_dir
            elif data_mode == "Link":
                path = self.input_external_link.get_value()

        return Import.Context(
            team_id=team_id,
            workspace_id=workspace_id,
            project_id=project_id,
            dataset_id=dataset_id,
            path=path,
            is_directory=is_directory,
            is_on_agent=is_on_agent,
            is_path_required=self.is_path_required(),
        )
