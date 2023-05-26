from os.path import basename, join
from typing import Optional, Union, List, Tuple, Dict, Any, Literal

import supervisely.io.env as env
from supervisely.project.project_type import ProjectType
from supervisely._utils import is_development, is_production
from supervisely.api.api import Api
from supervisely.app import get_data_dir, DialogWindowError
from supervisely.app.fastapi.subapp import Application
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    DatasetThumbnail,
    Empty,
    Field,
    FileStorageUpload,
    FileThumbnail,
    FolderThumbnail,
    Input,
    ProjectThumbnail,
    SelectDataset,
    SelectProject,
    SelectWorkspace,
    SlyTqdm,
    Stepper,
    Tabs,
    RadioTabs,
    TeamFilesSelector,
    Text,
    Widget,
)
from supervisely.sly_logger import logger
from tqdm import tqdm

# @TODO: set project_type in constructor                    +DONE+
# @TODO: flexible constructor (select tabs)                 +DONE+
# @TODO: destination project                                +DONE+
# @TODO: add stepper                                        +DONE+
# @TODO: add output card                                    +DONE+
# @TODO: Settings card with custom settings                 +DONE+
# @TODO: catch errors and show dialog message               +DONE+
# @TODO: add thumbnail for result project to output card    +DONE+
# @TODO: add confirm button and reselect                    +DONE+
# @TODO: add validation for each step                       +DONE+
# @TODO: add remove source to context                       +DONE+
# @TODO: add value_change to Tabs                           +DONE+

# @TODO: fix folder thumbnail - add .tmp file for link      +DONE+
# @TODO: add card lock messages
# @TODO: add input card with context

# @TODO: first card - select data (drag n drop by deffault)
# @TODO: Tabs -> RadioTabs
# @TODO: add data_type to constructor - only folders or onlny files?

PROJECT_TYPES: List[ProjectType] = [
    ProjectType.IMAGES,
    ProjectType.VIDEOS,
    ProjectType.VOLUMES,
    ProjectType.POINT_CLOUDS,
    ProjectType.POINT_CLOUD_EPISODES,
]
DESTINATION_OPTIONS = ["New Project", "Existing Project", "Existing Dataset"]
DATAS_ELECTION_OPTIONS = ["File Selector", "Drag & Drop"]


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
            remove_source_files: bool = False,
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

            self._path = path
            if self._path is None:
                raise ValueError(f"Remote path is not specified: {self._path}")
            if type(self._path) is not str:
                raise ValueError(f"Remote path must be 'str': {self._path}")

            self._is_directory = is_directory
            if type(self._is_directory) is not bool:
                raise ValueError(f"Flag 'is_directory' must be 'bool': {self._is_directory}")

            self._is_on_agent = is_on_agent
            if type(self._is_on_agent) is not bool:
                raise ValueError(f"Flag 'is_on_agent' must be 'bool': {self._is_on_agent}")

            self._remove_source_files = remove_source_files
            if type(self._is_on_agent) is not bool:
                raise ValueError(
                    f"Flag 'remove_source_files' must be 'bool': {self._remove_source_files}"
                )

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
                f"Project type: {self._project_type}\n"
                f"Path: {self._path}\n"
                f"Is directory: {self._is_directory}\n"
                f"Is on agent: {self._is_on_agent}\n"
                f"Remove source files: {self._remove_source}\n"
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

    def __init__(
        self,
        allowed_project_types: List[ProjectType] = None,
        allowed_destination_options: List[str] = None,
    ):
        if allowed_project_types is None:
            allowed_project_types = PROJECT_TYPES
            for option in allowed_project_types:
                if option not in PROJECT_TYPES:
                    raise ValueError(f"Project type must be one of {PROJECT_TYPES}: {option}")

            if len(allowed_project_types) > 5:
                raise ValueError(
                    f"Maximum 5 project types can be specified: {allowed_project_types}"
                )

        self._allowed_project_types: List[ProjectType] = allowed_project_types

        if allowed_destination_options is None:
            allowed_destination_options = DESTINATION_OPTIONS
            for option in allowed_destination_options:
                if option not in DESTINATION_OPTIONS:
                    raise ValueError(f"Destination must be one of {DESTINATION_OPTIONS}: {option}")

            if len(allowed_destination_options) > 3:
                raise ValueError(
                    f"Maximum 3 destinations can be specified: {allowed_destination_options}"
                )

        self._allowed_destination_options: Literal[
            "New Project", "Existing Project", "Existing Dataset"
        ] = allowed_destination_options

        self._layout = None
        if is_development():
            self.import_progress = tqdm

        if is_production():
            api = Api.from_env()

            self._step_one_btn = Button("Select", icon="zmdi zmdi-check")
            self._step_one_text = Text(status="error")
            self._step_one_text.hide()

            self._step_two_btn = Button("Select", icon="zmdi zmdi-check")
            self._step_two_text = Text(status="error")
            self._step_two_text.hide()

            self._step_three_btn = Button("Select", icon="zmdi zmdi-check")

            self.output_project_tabs = None
            self.input_data_tabs = None
            # self._step_four_btn = Button("Select")

            ####################
            # PROJECT SELECTOR #
            ####################

            self.input_mode = None

            # Launch from Dataset
            if (
                env.project_id(raise_not_found=False) is not None
                and env.dataset_id(raise_not_found=False) is not None
            ):
                project_info = api.project.get_info_by_id(env.project_id())
                dataset_info = api.dataset.get_info_by_id(env.dataset_id())
                self.input_thumbnail = DatasetThumbnail(
                    project_info=project_info, dataset_info=dataset_info
                )
                self.project_selector_card_widgets = Container(widgets=[self.input_thumbnail])

                self.input_mode = "dataset"

            # Launch from Project
            elif env.project_id(raise_not_found=False) is not None:
                project_info = api.project.get_info_by_id(env.project_id())
                self.input_thumbnail = ProjectThumbnail(info=project_info)
                self.project_selector_card_widgets = Container(widgets=[self.input_thumbnail])
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
                self.output_project_selector = SelectProject(
                    allowed_types=self._allowed_project_types
                )
                self.output_dataset_selector = SelectDataset(
                    allowed_project_types=self._allowed_project_types
                )
                destination_widgets_map = {
                    "New Project": self.output_new_project,
                    "Existing Project": self.output_project_selector,
                    "Existing Dataset": self.output_dataset_selector,
                }

                if len(self._allowed_destination_options) > 1:
                    self.output_project_tabs = Tabs(
                        labels=self._allowed_destination_options,
                        contents=[
                            destination_widgets_map[option]
                            for option in self._allowed_destination_options
                        ],
                    )
                    self.project_selector_card_widgets = Container(
                        widgets=[self.output_project_tabs, self._step_one_text]
                    )
                else:
                    self.project_selector_card_widgets = Container(
                        widgets=[
                            destination_widgets_map[self._allowed_destination_options[0]],
                            self._step_one_text,
                        ]
                    )
                self.input_mode = "ecosystem"

            ####################
            #  FILES SELECTOR  #
            ####################

            # Launch from File
            if env.file(raise_not_found=False) is not None:
                file_info = api.file.get_info_by_path(team_id=env.team_id(), remote_path=env.file())
                self.input_thumbnail = FileThumbnail(info=file_info)
                self.files_selector_card_widgets = Container(widgets=[self.input_thumbnail])
                if self.input_mode is None or self.input_mode == "ecosystem":
                    self.input_mode = "file"

            # Launch from Folder
            elif env.folder(raise_not_found=False) is not None:
                import tempfile

                temp_file = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
                temp_file.close()
                file_path = temp_file.name
                remote_path = join(env.folder(), basename(file_path))
                file_info = api.file.upload(team_id=env.team_id(), src=file_path, dst=remote_path)
                self.input_thumbnail = FolderThumbnail(info=file_info)
                api.file.remove(team_id=env.team_id(), path=remote_path)
                self.files_selector_card_widgets = Container(widgets=[self.input_thumbnail])
                if self.input_mode is None or self.input_mode == "ecosystem":
                    self.input_mode = "folder"

            # Launch from Ecosystem
            else:
                self.input_file_selector = TeamFilesSelector(
                    team_id=env.team_id(), multiple_selection=False, max_height=300
                )

                app_name = env.app_name(raise_not_found=False)
                if app_name is None:
                    app_name = "import-template"
                else:
                    app_name = app_name.lower().replace(" ", "-")
                self.storage_upload_path = f"/import/{app_name}/"
                self.input_drag_n_drop = FileStorageUpload(
                    team_id=env.team_id(), path=self.storage_upload_path
                )
                self.input_data_tabs = Tabs(
                    labels=["File Selector", "Drag & Drop"],
                    contents=[
                        self.input_file_selector,
                        self.input_drag_n_drop,
                    ],
                )
                self.files_selector_card_widgets = Container(
                    widgets=[self.input_data_tabs, self._step_two_text]
                )
                if self.input_mode is None:
                    self.input_mode = "ecosystem"

            ############
            # SETTINGS #
            #############

            self.remove_source_files_checkbox = Checkbox(
                "Remove source files after import", checked=True
            )
            self.settings_card_widgets = Container(widgets=[self.remove_source_files_checkbox])

            custom_settings_container = self.generate_custom_settings()
            if custom_settings_container is not None:
                custom_field = Field(content=custom_settings_container, title="Custom Settings")
                self.settings_card_widgets._widgets.append(custom_field)

            self.settings_card = Card(
                title="Import Settings",
                description="Configure import settings",
                content=self.settings_card_widgets,
            )

            ##########
            # OUTPUT #
            ##########

            self.output_project_thumbnail = ProjectThumbnail()
            self.output_project_thumbnail.hide()
            self._start_button = Button("Start Import")
            self.import_progress = SlyTqdm()
            self.output_card_container = Container(
                widgets=[self.output_project_thumbnail, self._start_button, self.import_progress]
            )
            self.output_card = Card(
                title="Output",
                description="Link to access result project will appear below after successful import",
                content=self.output_card_container,
            )

            ##########
            # LAYOUT #
            ##########

            if self.input_mode == "project":
                self.input_project_card = Card(
                    title="Destination Project",
                    description="Application was launched from the context menu of the project",
                    content=self.project_selector_card_widgets,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import. "
                        "You can select files from Team Files or use Drag & Drop"
                    ),
                    content=self.files_selector_card_widgets,
                )

                self.input_project_container = Container(widgets=[self.input_project_card])
                self.files_selector_container = Container(
                    widgets=[self.input_files_card, self._step_two_btn]
                )
                self.settings_container = Container(
                    widgets=[self.settings_card, self._step_three_btn]
                )

                self._layout = Stepper(
                    widgets=[
                        self.input_project_container,
                        self.files_selector_container,
                        self.settings_container,
                        self.output_card,
                    ],
                    active_step=2,
                )

            elif self.input_mode == "dataset":
                self.input_project_card = Card(
                    title="Destination Dataset",
                    description="Application was launched from the context menu of the dataset",
                    content=self.project_selector_card_widgets,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import. "
                        "You can select files from Team Files or use Drag & Drop"
                    ),
                    content=self.files_selector_card_widgets,
                )

                self.input_project_container = Container(widgets=[self.input_project_card])
                self.files_selector_container = Container(
                    widgets=[self.input_files_card, self._step_two_btn]
                )
                self.settings_container = Container(
                    widgets=[self.settings_card, self._step_three_btn]
                )

                self._layout = Stepper(
                    widgets=[
                        self.input_project_container,
                        self.files_selector_container,
                        self.settings_container,
                        self.output_card,
                    ],
                    active_step=2,
                )

            elif self.input_mode == "file":
                self.input_project_card = Card(
                    title="Destination Project",
                    description="Select where do you want to import your data",
                    content=self.project_selector_card_widgets,
                )
                self.input_files_card = Card(
                    title="Input File",
                    description="Application was launched from the context menu of the file",
                    content=self.files_selector_card_widgets,
                )

                self.files_selector_container = Container(widgets=[self.input_files_card])
                self.input_project_container = Container(
                    widgets=[self.input_project_card, self._step_two_btn]
                )

                self.settings_container = Container(
                    widgets=[self.settings_card, self._step_three_btn]
                )

                self._layout = Stepper(
                    widgets=[
                        self.files_selector_container,
                        self.input_project_container,
                        self.settings_container,
                        self.output_card,
                    ],
                    active_step=2,
                )

            elif self.input_mode == "folder":
                self.input_project_card = Card(
                    title="Destination Project",
                    description="Select where do you want to import your data",
                    content=self.project_selector_card_widgets,
                )
                self.input_files_card = Card(
                    title="Input Folder",
                    description="Application was launched from the context menu of the folder",
                    content=self.files_selector_card_widgets,
                )

                self.files_selector_container = Container(widgets=[self.input_files_card])
                self.input_project_container = Container(
                    widgets=[self.input_project_card, self._step_two_btn]
                )

                self.settings_container = Container(
                    widgets=[self.settings_card, self._step_three_btn]
                )

                self._layout = Stepper(
                    widgets=[
                        self.files_selector_container,
                        self.input_project_container,
                        self.settings_container,
                        self.output_card,
                    ],
                    active_step=2,
                )
            else:
                self.input_project_card = Card(
                    title="Destination Project",
                    description="Select where do you want to import your data",
                    content=self.project_selector_card_widgets,
                )
                self.input_files_card = Card(
                    title="Select Files",
                    description=(
                        "Select data to import. "
                        "You can select files from Team Files or use Drag & Drop"
                    ),
                    content=self.files_selector_card_widgets,
                )

                self.input_project_container = Container(
                    widgets=[self.input_project_card, self._step_one_btn]
                )
                self.files_selector_container = Container(
                    widgets=[self.input_files_card, self._step_two_btn]
                )
                self.settings_container = Container(
                    widgets=[self.settings_card, self._step_three_btn]
                )

                self._layout = Stepper(
                    widgets=[
                        self.input_project_container,
                        self.files_selector_container,
                        self.settings_container,
                        self.output_card,
                    ],
                    active_step=1,
                )

            self.settings_card.lock()
            self.output_card.lock()
        super().__init__(layout=self._layout)

    def __get_context(self, api: Api) -> Context:
        team_id = env.team_id()
        workspace_id = env.workspace_id()
        project_id = None
        dataset_id = None
        path = None
        is_directory = True
        is_on_agent = False
        remove_source_files = False

        if is_production():
            if self.input_mode == "project":
                project_id = env.project_id()
            elif self.input_mode == "dataset":
                project_id = env.project_id()
                dataset_id = env.dataset_id()
            elif self.input_mode in ["ecosystem", "file", "folder"]:
                project_mode = self.output_project_tabs.get_active_tab()
                if project_mode == "New Project":
                    team_id = self.output_workspace_selector._team_selector.get_selected_id()
                    workspace_id = self.output_workspace_selector.get_selected_id()
                    project_name = self.output_new_project_name.get_value()
                elif project_mode == "Existing Project":
                    team_id = (
                        self.output_project_selector._ws_selector._team_selector.get_selected_id()
                    )
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

            data_dir = get_data_dir()
            if self.input_mode == "file":
                path = env.file()
                local_save_path = join(data_dir, basename(path.rstrip("/")))
                api.file.download(team_id=team_id, remote_path=path, local_save_path=path)
                is_directory = False
                path = local_save_path
            elif self.input_mode == "folder":
                path = env.folder()
                local_save_path = join(data_dir, basename(path.rstrip("/")))
                api.file.download_directory(
                    team_id=team_id, remote_path=path, local_save_path=local_save_path
                )
                is_directory = True
                path = local_save_path
            elif self.input_mode in ["ecosystem", "project", "dataset"]:
                data_mode = self.input_data_tabs.get_active_tab()
                if data_mode == "File Selector":
                    paths = self.input_file_selector.get_selected_paths()

                    data_path = self.input_file_selector.get_selected_paths()[0]
                    data_type = self.input_file_selector.get_selected_items()[0]["type"]

                    path = join(data_dir, basename(data_path.rstrip("/")))
                    if data_type == "file":
                        api.file.download(
                            team_id=team_id, remote_path=data_path, local_save_path=path
                        )
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

            remove_source_files = self.remove_source_files_checkbox.is_checked()
            return Import.Context(
                team_id=team_id,
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=path,
                is_directory=is_directory,
                is_on_agent=is_on_agent,
                remove_source_files=remove_source_files,
            )

        # if is_development():
        else:
            file = env.file(raise_not_found=False)
            folder = env.folder(raise_not_found=False)

            if file is not None and folder is not None:
                raise KeyError(
                    "Both FILE and FOLDER envs are defined, but only one is allowed for the import"
                )

            if file is None and folder is None:
                raise KeyError(
                    "One of the environment variables has to be defined for the import app: FILE or FOLDER"
                )

            path = folder
            if file is not None:
                path = file
                is_directory = False

            is_on_agent = False
            if path is not None:
                is_on_agent = api.file.is_on_agent(path)

            project_id = env.project_id(raise_not_found=False)
            dataset_id = env.dataset_id(raise_not_found=False)
            if project_id is not None:
                project = api.project.get_info_by_id(id=project_id)
                if project is None:
                    raise ValueError(
                        f"Project with ID: '{project_id}' is not found or either archived"
                    )
                logger.info(f"Importing to existing Project: id={project.id}, name={project.name}")

            if dataset_id is not None:
                dataset = api.dataset.get_info_by_id(id=dataset_id)
                if dataset is None:
                    raise ValueError(
                        f"Dataset with ID: '{dataset_id}' is not found or either archived"
                    )
                logger.info(f"Importing to existing Dataset: id={dataset.id}, name={dataset.name}")

            return Import.Context(
                team_id=team_id,
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=path,
                is_directory=is_directory,
                is_on_agent=is_on_agent,
            )

    def generate_custom_settings(self) -> List[Widget]:
        # implement your own method for import settings
        return None

    def process(self, context: Context) -> Optional[Union[int, None]]:
        # implement your own method for import
        raise NotImplementedError()

    def run(self):
        api = Api.from_env()
        task_id = None

        if is_development():
            context = self.__get_context(api)
            project_id = self.process(context)
            info = api.project.get_info_by_id(project_id)
            logger.info(f"Result project: id={info.id}, name={info.name}")

        if is_production() is True:
            task_id = env.task_id()

            if self.output_project_tabs is not None:

                @self.output_project_tabs.click
                def hide_message_project(value):
                    self._step_one_text.hide()

            if self.input_data_tabs is not None:

                @self.input_data_tabs.click
                def hide_message_files(value):
                    self._step_two_text.hide()

            @self._step_one_btn.click
            def finish_step_one():
                if (
                    self.output_project_tabs is not None
                    and len(self._allowed_destination_options) > 1
                ):
                    if self.output_project_tabs.get_active_tab() == "New Project":
                        if self.output_new_project_name.get_value() == "":
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please enter a project name", status="error"
                            )
                            return
                    elif self.output_project_tabs.get_active_tab() == "Existing Project":
                        if self.output_project_selector.get_selected_id() is None:
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please select a project to import to", status="error"
                            )
                            return
                    elif self.output_project_tabs.get_active_tab() == "Existing Dataset":
                        if (
                            self.output_dataset_selector.get_selected_id() is None
                            or len(self.output_dataset_selector.get_selected_id()) == 0
                        ):
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please select a dataset to import to", status="error"
                            )
                            return

                else:
                    if self._allowed_destination_options == ["New Project"]:
                        if self.output_new_project_name.get_value() == "":
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please enter a project name", status="error"
                            )
                            return
                    elif self._allowed_destination_options == ["Existing Project"]:
                        if self.output_project_selector.get_selected_id() is None:
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please select a project to import to", status="error"
                            )
                            return
                    elif self._allowed_destination_options == ["Existing Dataset"]:
                        if (
                            self.output_dataset_selector.get_selected_id() is None
                            or len(self.output_dataset_selector.get_selected_id()) == 0
                        ):
                            self._step_one_text.show()
                            self._step_one_text.set(
                                text="Please select a dataset to import to", status="error"
                            )
                            return

                if self._step_one_btn.text == "Select":
                    self._step_one_text.hide()
                    self._step_one_btn.text = "Reselect"
                    self._step_one_btn.icon = "zmdi zmdi-edit"
                    self.input_project_card.lock()
                    self.input_files_card.unlock()
                    self._layout.set_active_step(2)

                elif self._step_one_btn.text == "Reselect":
                    self._step_one_text.hide()
                    self._step_one_btn.text = "Select"
                    self._step_one_btn.icon = "zmdi zmdi-check"
                    self.input_project_card.unlock()
                    self.input_files_card.lock()
                    self.settings_card.lock()
                    self.output_card.lock()
                    self._layout.set_active_step(1)

            @self._step_two_btn.click
            def finish_step_two():
                if self.input_mode in ["file", "folder"]:
                    if self.output_project_tabs is not None:
                        if self.output_project_tabs.get_active_tab() == "New Project":
                            if self.output_new_project_name.get_value() == "":
                                self._step_two_text.show()
                                self._step_two_text.set(
                                    text="Please enter a project name", status="error"
                                )
                                return
                        elif self.output_project_tabs.get_active_tab() == "Existing Project":
                            if self.output_project_selector.get_selected_id() is None:
                                self._step_two_text.show()
                                self._step_two_text.set(
                                    text="Please select a project to import to", status="error"
                                )
                                return
                        elif self.output_project_tabs.get_active_tab() == "Existing Dataset":
                            if self.output_dataset_selector.get_selected_id() is None:
                                self._step_two_text.show()
                                self._step_two_text.set(
                                    text="Please select a dataset to import to", status="error"
                                )
                                return

                elif self.input_mode in ["ecosystem", "project", "dataset"]:
                    # "File Selector", "Drag & Drop"
                    if self.input_data_tabs is not None:
                        if self.input_data_tabs.get_active_tab() == "File Selector":
                            if len(self.input_file_selector.get_selected_items()) == 0:
                                self._step_two_text.show()
                                self._step_two_text.set(
                                    text="Please select a data to import", status="error"
                                )
                                return
                        elif self.input_data_tabs.get_active_tab() == "Drag & Drop":
                            if len(self.input_drag_n_drop.get_uploaded_paths()) == 0:
                                self._step_two_text.show()
                                self._step_two_text.set(
                                    text="Please drag & drop data to import", status="error"
                                )
                                return

                if self._step_two_btn.text == "Select":
                    self._step_two_text.hide()
                    self._step_two_btn.text = "Reselect"
                    self._step_two_btn.icon = "zmdi zmdi-edit"
                    self.input_project_card.lock()
                    self.input_files_card.lock()
                    self.settings_card.unlock()
                    self.output_card.lock()
                    self._layout.set_active_step(3)

                elif self._step_two_btn.text == "Reselect":
                    self._step_two_text.hide()
                    self._step_two_btn.text = "Select"
                    self._step_two_btn.icon = "zmdi zmdi-check"
                    self.input_project_card.lock()
                    self.input_files_card.unlock()
                    self.settings_card.lock()
                    self.output_card.lock()
                    self._layout.set_active_step(2)

            @self._step_three_btn.click
            def finish_step_three():
                if self._step_three_btn.text == "Select":
                    self._step_three_btn.text = "Reselect"
                    self._step_three_btn.icon = "zmdi zmdi-edit"
                    self.input_project_card.lock()
                    self.input_files_card.lock()
                    self.settings_card.lock()
                    self.output_card.unlock()
                    self._layout.set_active_step(4)

                elif self._step_three_btn.text == "Reselect":
                    self._step_three_btn.text = "Select"
                    self._step_three_btn.icon = "zmdi zmdi-check"
                    self.input_project_card.lock()
                    self.input_files_card.lock()
                    self.settings_card.unlock()
                    self.output_card.lock()
                    self._layout.set_active_step(3)

            @self._start_button.click
            def start_import():
                try:
                    self._step_one_btn.disable()
                    self._step_two_btn.disable()
                    self._step_three_btn.disable()
                    context = self.__get_context(api)
                    self.import_progress.show()
                    project_id = self.process(context)
                    if type(project_id) is int:
                        info = api.project.get_info_by_id(project_id)
                        api.task.set_output_project(task_id, info.id, info.name)
                        logger.info(f"Result project: id={info.id}, name={info.name}")
                        self.output_project_thumbnail.set(info)
                        self.output_project_thumbnail.show()
                    else:
                        self._step_one_btn.enable()
                        self._step_two_btn.enable()
                        self._step_three_btn.enable()
                        raise DialogWindowError(
                            title="Import error",
                            description=f"Error: '{project_id}' is not a valid project id",
                        )

                    self._start_button.disable()
                except Exception as e:
                    self.import_progress.hide()
                    self.output_project_thumbnail.hide()
                    self._start_button.enable()
                    self._start_button.loading = False
                    raise DialogWindowError(title="Import error", description=f"Error: {str(e)}")
