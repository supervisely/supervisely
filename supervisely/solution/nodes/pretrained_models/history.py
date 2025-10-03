from typing import List

from supervisely.api.api import Api
from supervisely.solution.components import TasksHistoryWidget


class PretrainedModelsTasksHistory(TasksHistoryWidget):

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Model ID",
                "Started At",
                "Status",
                "Agent ID",
                "Classes Count",
                "Images Count",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["task_id"],
                ["model_id"],
                ["task_info", "startedAt"],
                ["status"],
                ["agent_id"],
                ["classes_count"],
                ["images_count"],
            ]
        return self._columns_keys

    def update_task_status(self, task_id: int, status: str):
        tasks = self.get_tasks()
        task = None
        for row in tasks:
            if row["id"] == task_id:
                task = row
                row["status"] = status
                self.update_task(task_id=task_id, task=task)
                return
        raise KeyError(f"Task with ID {task_id} not found in the task history.")
