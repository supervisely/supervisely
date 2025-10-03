from typing import List

from supervisely.solution.components import TasksHistoryWidget


class DeployCustomModelHistory(TasksHistoryWidget):

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Model Name",
                "Started At",
                "Hardware",
                "Device",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["id"],
                ["model_name"],
                ["started_at"],
                ["hardware"],
                ["device"],
            ]
        return self._columns_keys

    def update(self):
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

    def add_task(self, task):
        super().add_task(task)
        self.update()
        self.update()
