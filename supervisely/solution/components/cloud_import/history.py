from supervisely.api.api import Api
from supervisely.solution.components.base import BaseHistory


class CloudImportHistory(BaseHistory):
    def __init__(self, api: Api, title: str = "Tasks History"):
        super().__init__(api, title)
        self.tasks_history.table_columns = [
            "Task ID",
            "Path",
            "Items Count",
            "Started At",
            "Status",
        ]
        self.tasks_history.columns_keys = [
            ["task_info", "id"],
            ["path"],
            ["items_count"],
            ["task_info", "created_at"],
            ["status"],
        ]
