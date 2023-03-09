from typing import List, Optional, Dict
from supervisely.app import StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class TaskLogs(Widget):
    def __init__(
        self,
        task_id: int,
        multiple: bool = False,
        filterable: bool = True,
        widget_id: str = None,
    ):
        self._task_id = task_id
        self._multiple = multiple
        self._filterable = filterable

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "taskId": self._task_id,
            "options": {"multiple": self._multiple, "filterable": self._filterable},
        }

    def get_json_state(self) -> Dict:
        return {}
