from typing import List

from supervisely.api.api import Api
from supervisely.solution.components import TasksHistoryWidget


class CloudImportTasksHistory(TasksHistoryWidget):

    def __init__(self, api: Api, project_id: int, *args, **kwargs):
        self.project_id = project_id
        self.api = api
        super().__init__(*args, **kwargs)

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "App Name",
                "Dataset ID",
                "Created At",
                "Images Count",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["id"],
                ["meta", "app", "name"],
                ["dataset_ids"],
                ["startedAt"],
                ["items_count"],
                ["status"],
            ]
        return self._columns_keys

    def update(self):
        tasks = self.get_tasks().copy()
        project = self.api.project.get_info_by_id(self.project_id)
        full_history = project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        for task in tasks:
            task_id = task["id"]
            history_item = history_dict.get(task_id)
            if history_item is None:
                task["dataset_ids"] = ""
                task["timestamp"] = ""
                task["items_count"] = 0
            else:
                datasets = history_item.get("datasets", [])
                task["dataset_ids"] = ", ".join(str(d["id"]) for d in datasets)
                task["timestamp"] = history_item.get("timestamp", "")
                task["items_count"] = history_item.get("items_count", 0)
            self.update_task(task_id, task)
        super().update()
