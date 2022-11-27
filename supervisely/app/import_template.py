from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env
from supervisely.api.api import Api

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Import:
    def process(
        self, workspace_id: int, path: str, is_directory: bool
    ) -> Optional[Union[int, None]]:
        pass

    def run(self):
        task_id = None
        if is_production():
            task_id = env.task_id()
            raise NotImplementedError()

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
        if file is not None:
            is_directory = False

        project_id = self.process(workspace_id=workspace_id, path=file, is_directory=is_directory)
        if type(project_id) is int and is_production():
            api = Api.from_env()
            info = api.project.get_info_by_id(project_id)
            api.task.set_output_project(task_id=task_id, project_id=info.id, project_name=info.name)
            print("Result project: id={info.id}, name={info.name}")
