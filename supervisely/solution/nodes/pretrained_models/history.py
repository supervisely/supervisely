from supervisely.api.api import Api
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory


class PretrainedModelsTasksHistory(TasksHistory):
    def __init__(self, api: Api, title: str = "Tasks History"):
        super().__init__(api, title)
        self._table_columns = [
            "Task ID",
            "Model ID",
            "Started At",
            "Status",
            "Agent ID",
            "Classes Count",
            "Images Count",
        ]
        self._columns_keys = [
            ["task_id"],
            ["model_id"],
            ["task_info", "startedAt"],
            ["status"],
            ["agent_id"],
            ["classes_count"],
            ["images_count"],
        ]

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
