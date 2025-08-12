from typing import Any, Dict, List, Optional

from supervisely.api.api import Api
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory


class MoveLabeledTasksHistory(TasksHistory):
    """Tasks history widget specialised for MoveLabeled node."""

    def __init__(self, widget_id: str = None):
        super().__init__(widget_id=widget_id)
        self._table_columns = [
            "Task ID",
            "Started At",
            "Images Count",
            "Status",
        ]
        self._columns_keys = [
            ["id"],
            ["startedAt"],
            ["images_count"],
            ["status"],
        ]

