from supervisely.api.api import Api
from supervisely.solution.components.base import BaseHistory


class ManualImportHistory(BaseHistory):
    """History table for Manual Import tasks."""

    def __init__(self, api: Api, title: str = "Tasks History"):
        super().__init__(api, title)
        self.tasks_history.table_columns = [
            "Task ID",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        self.tasks_history.columns_keys = [
            ["task_info", "id"],
            ["datasets", "id"],
            ["task_info", "created_at"],
            ["items_count"],
            ["status"],
        ]
