from typing import Any, Dict, List

from supervisely.solution.components import TasksHistoryWidget


class PreLabelingTasksHistory(TasksHistoryWidget):

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Started At",
                "Images Count",
                "Status",
                "Duration",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["task_id"],
                ["started_at"],
                ["images_count"],
                ["status"],
                ["duration"],
            ]
        return self._columns_keys

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Dict[str, Any]) -> int:
        super().add_task(task)
        self.update()
