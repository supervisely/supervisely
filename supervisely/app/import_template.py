from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from os.path import join, basename
from pathlib import Path

from typing import NamedTuple

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Import:
    class Context:
        def __init__(self, team_id: int, workspace_id: int, project_id: int, dataset_id: int, path: str, is_directory: bool = True):            
            self.team_id = team_id
            if self.team_id is None:
                raise ValueError(f"Team ID is not specified: {self.team_id}")
            if type(self.team_id) != int:
                raise ValueError(f"Project ID must be 'int': {self.project_id}")

            self.workspace_id = workspace_id
            if self.workspace_id is None:
                raise ValueError(f"Workspace ID is not specified: {self.workspace_id}")
            if type(self.workspace_id) != int:
                raise ValueError(f"Project ID must be 'int': {self.project_id}")

            self.project_id = project_id
            if self.project_id is None:
                raise ValueError(f"Project ID is not specified: {self.project_id}")
            if type(self.project_id) != int:
                raise ValueError(f"Project ID must be 'int': {self.project_id}")

            self.dataset_id = dataset_id
            if self.project_id is None:
                raise ValueError(f"Dataset ID is not specified: {self.dataset_id}")
            if type(self.dataset_id) != int:
                raise ValueError(f"Dataset ID must be 'int': {self.dataset_id}")

            self.path = path
            if self.path is None:
                raise ValueError(f"Remote path to files is not specified: {self.path}")
            if type(self.path) != str:
                raise ValueError(f"Remote path must be 'str': {self.path}")

            self.is_directory = is_directory
            if type(self.is_directory) != bool:
                raise ValueError(f"Remote path must be 'bool': {self.is_directory}")
       
        def print_context(self):
            if self.team_id is not None:
                print(f"'team id:' {self.team_id}")
            if self.workspace_id is not None:
                print(f"'workspace id:' {self.workspace_id}")
            if self.project_id is not None:
                print(f"'Items will be uploaded to project:' {self.project_id}")
            if self.dataset_id is not None:
                print(f"'dataset id:' {self.dataset_id}")
            if self.path is not None:
                print(f"path:' {self.path}")
            if self.is_directory is not None:
                print(f"is_directory:' {self.is_directory}")
        
        def print_destination(self):
            if self.team_id is not None and self.workspace_id is not None and self.project_id is not None:
                if self.dataset_id is not None:
                    print(f"Files will be uploaded to the following destination:"
                          f"\nteam id: {self.team_id}"
                          f"\nworkspace id: {self.workspace_id}"
                          f"\nproject id: {self.project_id}"
                          f"\ndataset id: {self.dataset_id}")
                else:
                    print(f"Files will be uploaded to the following destination:"
                          f"\nteam id: {self.team_id}"
                          f"\nworkspace id: {self.workspace_id}"
                          f"\nproject id: {self.project_id}")
                
        def print_remote_path(self):
            if self.path is not None:
                if self.is_directory is True:
                    print(f"Remote path to directory:' {self.path}")
                else:
                    print(f"Remote path to file:' {self.path}")


        
    def process(self, context: Context) -> Optional[Union[int, None]]:
        raise NotImplementedError() # implement your own method when inherit

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
        
        # get or create project with the same name as input file and empty dataset in it
        if project_id is None:
            project_name = Path(path).stem
            project = api.project.create(workspace_id=workspace_id, name=project_name, change_name_if_conflict=True)
        else:
            project = api.project.get_info_by_id(id=project_id)
        print(f"Working project: id={project.id}, name={project.name}")
            
        if dataset_id is None: 
            dataset = api.dataset.create(project_id=project.id, name="dataset", change_name_if_conflict=True)
        else:
            dataset = api.dataset.get_info_by_id(id=dataset_id)
        print(f"Working dataset: id={dataset.id}, name={dataset.name}")

        if is_production():
            local_save_path = join(get_data_dir(), basename(path))
            if is_directory:
                api.file.download_directory(team_id=team_id, remote_path=path, local_save_path=local_save_path)
            else:
                api.file.download(team_id=team_id, remote_path=path, local_save_path=local_save_path)
            path = local_save_path
            
        context = Import.Context(
            team_id=team_id,
            workspace_id=workspace_id,
            project_id=project.id,
            dataset_id=dataset.id,
            path=path,
            is_directory=is_directory
        )
            
        project_id = self.process(context=context)
        if type(project_id) is int and is_production():
            info = api.project.get_info_by_id(project_id)
            api.task.set_output_project(task_id=task_id, project_id=info.id, project_name=info.name)
            print(f"Result project: id={info.id}, name={info.name}")
