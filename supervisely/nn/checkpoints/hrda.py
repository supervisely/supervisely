from os.path import basename, join
from typing import List

from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely.io.json import load_json_file
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


# not enough info to implement
def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    raise NotImplementedError
