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
    class Context(NamedTuple):
        task_id=None
        team_id=None
        workspace_id=None
        project_id=None
        dataset_id=None
        path=None
        is_directory=True
    
    def process(self, context: Context) -> Optional[Union[int, None]]:
        raise NotImplementedError() # implement your own method when inherit

    def run(self):
        api = Api.from_env()
        context = Import.Context
        
        context.task_id = None
        if is_production():
            context.task_id = env.task_id()

        context.team_id = env.team_id()
        context.workspace_id = env.workspace_id()
          
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

        context.path = folder
        if file is not None:
            context.path = file
            context.is_directory = False
            
        context.project_id = env.project_id(raise_not_found=False)
        context.dataset_id = env.dataset_id(raise_not_found=False)
        
        # get or create project with the same name as input file and empty dataset in it
        if context.project_id is None:
            project_name = Path(context.path).stem
            project = api.project.create(workspace_id=context.workspace_id, name=project_name, change_name_if_conflict=True)
        else:
            project = api.project.get_info_by_id(id=project_id)
        print(f"Working project: id={project.id}, name={project.name}")
            
        if context.dataset_id is None: 
            dataset = api.dataset.create(project_id=project.id, name="dataset", change_name_if_conflict=True)
        else:
            dataset = api.dataset.get_info_by_id(id=context.dataset_id)
        print(f"Working dataset: id={dataset.id}, name={dataset.name}")

        if is_production():
            local_save_path = join(get_data_dir(), basename(path))
            if context.is_directory:
                raise NotImplementedError()
                # api.file.download_directory(team_id=team_id, remote_path=path, local_save_path=local_save_path)
            else:
                api.file.download(team_id=context.team_id, remote_path=path, local_save_path=local_save_path)
            path = local_save_path
            
        project_id = self.process(context=context)
        if type(project_id) is int and is_production():
            info = api.project.get_info_by_id(project_id)
            api.task.set_output_project(task_id=context.task_id, project_id=info.id, project_name=info.name)
            print(f"Result project: id={info.id}, name={info.name}")
