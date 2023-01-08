from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, basename
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Import:
    class Context:
        def __init__(
            self,
            team_id: int,
            workspace_id: int,
            project_id: int,
            dataset_id: int,
            path: str,
            is_directory: bool = True,
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
            self._path = path
            if self._is_path_required is True and self._path is None:
                raise ValueError(f"Remote path is not specified: {self._path}")
            if self._is_path_required is True and type(self._path) is not str:
                raise ValueError(f"Remote path must be 'str': {self._path}")

            self._is_directory = is_directory
            if self._is_path_required is True and type(self._is_directory) is not bool:
                raise ValueError(f"Remote path must be 'bool': {self._is_directory}")

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
                f"Path: {self._path}\n"
                f"Is directory: {self._is_directory}"
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

    def process(self, context: Context) -> Optional[Union[int, None]]:
        raise NotImplementedError()  # implement your own method when inherit

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
        if self.is_path_required():
            if file is None and folder is None:
                raise KeyError(
                    "One of the environment variables has to be defined for the import app: FILE or FOLDER"
                )

        is_directory = True
        path = folder
        if file is not None:
            path = file
            is_directory = False

        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        if project_id is not None:
            # lets validate that project exists
            project = api.project.get_info_by_id(id=project_id)
            print(f"Importing to existing Project: id={project.id}, name={project.name}")
        if dataset_id is not None:
            # lets validate that dataset exists
            dataset = api.dataset.get_info_by_id(id=dataset_id)
            print(f"Importing to existing Dataset: id={dataset.id}, name={dataset.name}")

        if is_production():
            if path is not None:
                local_save_path = join(get_data_dir(), basename(path.rstrip("/")))
                if is_directory:
                    api.file.download_directory(
                        team_id=team_id, remote_path=path, local_save_path=local_save_path
                    )
                else:
                    api.file.download(
                        team_id=team_id, remote_path=path, local_save_path=local_save_path
                    )
                path = local_save_path

        context = Import.Context(
            team_id=team_id,
            workspace_id=workspace_id,
            project_id=project_id,
            dataset_id=dataset_id,
            path=path,
            is_directory=is_directory,
            is_path_required=self.is_path_required(),
        )

        project_id = self.process(context=context)
        if type(project_id) is int and is_production():
            info = api.project.get_info_by_id(project_id)
            api.task.set_output_project(task_id=task_id, project_id=info.id, project_name=info.name)
            print(f"Result project: id={info.id}, name={info.name}")
