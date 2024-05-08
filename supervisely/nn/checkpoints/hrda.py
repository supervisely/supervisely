from typing import List
from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


# not enough info to implement
def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    raise NotImplementedError
