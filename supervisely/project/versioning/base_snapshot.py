from typing import Callable, List, Optional, Protocol, Union

from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo

class BaseSnapshotHandler(Protocol):
    schema_version: str

    def build_payload(
        self,
        api: Api,
        project_id: int,
        payload_dir: str,
        dataset_ids: Optional[List[int]] = None,
        batch_size: int = 50,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None: ...

    def restore_payload(
        self,
        api: Api,
        payload_dir: str,
        workspace_id: int,
        project_name: Optional[str] = None,
        with_custom_data: bool = True,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed: bool = False,
    ) -> ProjectInfo: ...