from typing import Any, Dict

from supervisely.app.widgets import TasksHistory, Button


class PreLabelingTasksHistory(TasksHistory):
    def __init__(self, widget_id: str = None):
        super().__init__(widget_id=widget_id)
        self._table_columns = [
            "Task ID",
            "Started At",
            "Images Count",
            "Status",
            "Duration",
        ]
        self._columns_keys = [
            ["task_id"],
            ["started_at"],
            ["images_count"],
            ["status"],
            ["duration"],
        ]

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Dict[str, Any]) -> int:
        super().add_task(task)
        self.update()
