from typing import Any, List, Optional
from venv import logger

from supervisely.api.api import Api
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory
from supervisely.solution.components import TasksHistoryWidget


class AutoImportTasksHistory(TasksHistoryWidget):
    """Tasks history widget specialised for Auto Import node."""

    APP_SLUG = "supervisely-ecosystem/main-import"

    def __init__(
        self,
        api: Optional[Api] = None,
        project_id: Optional[str] = None,
        widget_id: Optional[str] = None,
    ):
        super().__init__(api, widget_id)
        self.project_id = project_id
        self._tasks: list[int] = []

    # ------------------------------------------------------------------
    # Table ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "App Name",
                "Dataset IDs",
                "Created At",
                "Images Count",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["task_id"],
                ["app", "name"],
                ["datasets", "id"],
                ["timestamp"],
                ["items_count"],
                ["status"],
            ]
        return self._columns_keys

    def _get_table_data(self) -> List[List]:
        """
        Collects and returns the import tasks history as a list of lists.
        """
        project = self.api.project.get_info_by_id(self.project_id)
        full_history = project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        for task in full_history:
            if task.get("slug") == self.APP_SLUG:
                task_id = task.get("task_id")
                if task_id is not None and task_id not in self._tasks:
                    self._tasks.append(task_id)

        data = []
        for task_id in self._tasks:
            history_item = history_dict.get(task_id)
            if history_item is None:
                data.append([task_id, "", "", "", 0, "failed"])
                continue
            if history_item.get("slug") != self.APP_SLUG:
                logger.warning(
                    f"Import history item with task_id {task_id} does not match the slug {self.APP_SLUG}. Skipping."
                )
                continue
            datasets = history_item.get("datasets", [])
            ds_ids = ", ".join(str(d["id"]) for d in datasets)
            status = history_item.get("status")
            if status == "started":
                status = "success"
            row = [
                history_item.get("task_id"),
                history_item.get("app", {}).get("name", ""),
                ds_ids,
                history_item.get("timestamp"),
                history_item.get("items_count"),
                status,
            ]
            data.append(row)

        return data

    def update(self):
        """Refresh the table with the current set of tasks."""
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)
