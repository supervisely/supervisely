from typing import Any, Dict

from supervisely.app.widgets import Button
from supervisely.solution.components import TasksHistoryWidget


class ComparisonHistory(TasksHistoryWidget):

    def __init__(
        self,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id)
        self._table_columns = [
            "Task ID",
            "Created At",
            "Comparison Report",
            "Best checkpoint",
        ]
        self._columns_keys = [
            ["id"],
            ["created_at"],
            ["result_link"],
            ["best_checkpoint"],
        ]

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Dict[str, Any]) -> int:
        super().add_task(task)
        self.update()
