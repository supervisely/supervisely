from typing import Optional, Union
from supervisely._utils import is_production
import supervisely.io.env as env

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal


class Import:
    def __init__(
        self,
    ):
        pass

    def process_file(self, workspace_id: int, path: str) -> Optional[Union[int, None]]:
        pass

    def process_folder(self, workspace_id: int, path: str) -> Optional[Union[int, None]]:
        pass

    def run(self):
        if is_production():
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

        if file is not None:
            result = self.process_file(workspace_id=workspace_id, path=file)
            print("result = ", result)
