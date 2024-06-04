from typing import List, Literal
from supervisely.api.api import Api
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo, Checkpoint


class HRDACheckpoint:
    def __init__(self, team_id: int):
        super().__init__(team_id)
        
        self._training_app = "Train HRDA"
        self._model_dir = "/HRDA"
        self._weights_dir = None
        self._task_type = "semantic segmentation"
        self._weights_ext = ".pth"
        self._config_file = "config.py"

    def get_list(self, sort: Literal["desc", "asc"] = "desc") -> List[CheckpointInfo]:
        # not enough info to implement
        raise NotImplementedError
