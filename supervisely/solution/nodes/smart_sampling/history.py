from typing import List, Optional, Any

from supervisely.api.api import Api
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory


class SmartSamplingTasksHistory(TasksHistory):
    """Tasks history widget specialised for Smart Sampling node."""

    def __init__(self, api: Optional[Api] = None, widget_id: Optional[str] = None):
        super().__init__(api, widget_id)

    # ------------------------------------------------------------------
    # Table ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "#",
                "Mode",
                "Date and Time",
                "Items Count",
                "Settings",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["mode"],
                ["timestamp"],
                ["items_count"],
                ["settings"],
                ["status"],
            ]
        return self._columns_keys

    def _get_table_data(self) -> List[List[Any]]:
        """Convert internal task dicts into table rows.

        Adds a running index as the first column so that rows match the
        :pyattr:`table_columns` definition.
        """
        tasks = self.get_tasks()
        rows: List[List[Any]] = []
        for idx, task in enumerate(tasks, start=1):
            settings = task.get("settings")
            rows.append(
                [
                    idx,
                    task.get("mode", "unknown"),
                    task.get("timestamp", "unknown"),
                    task.get("items_count", "unknown"),
                    str(settings) if settings else "-",
                    task.get("status", "unknown"),
                ]
            )
        return rows

    def update(self):
        """Refresh the table with the current set of tasks."""
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)
