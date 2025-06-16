import threading
import time
from typing import Any, Dict, List, Union
from venv import logger

from supervisely.api.api import Api
from supervisely.app import DataJson
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.app.widgets.task_logs.task_logs import TaskLogs
from supervisely.app.widgets.widget import Widget

_MISSING = object()


class TasksHistory(Widget):
    def __init__(
        self,
        api: Api = None,
        widget_id: str = None,
    ):
        self.api = api or Api()
        self._stop_autorefresh = False
        self._refresh_thread = None
        self._refresh_interval = 60
        super().__init__(widget_id=widget_id)

    @property
    def table_columns(self) -> List[str]:
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

    @table_columns.setter
    def table_columns(self, value: List[str]):
        self._table_columns = value

    def _get_task_item(
        self, keys: Union[str, int, List[Union[str, int]]], task: Dict[str, Any], default=_MISSING
    ) -> Any:
        if not isinstance(keys, list):
            keys = [keys]

        try:
            if len(keys) == 0:
                raise ValueError("keys must be a non-empty list")
            item = task
            for key in keys:
                item = item[key]
            return item
        except (KeyError, TypeError, IndexError, ValueError) as e:
            if default is not _MISSING:
                return default
            raise

    @property
    def columns_keys(self) -> List[List[str]]:
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["id"],
                ["meta", "app", "name"],
                ["dataset_ids"],
                ["created_at"],
                ["images_count"],
                ["status"],
            ]
        return self._columns_keys

    @columns_keys.setter
    def columns_keys(self, value: List[List[str]]):
        if not any(["id" in col or ["id"] in col for col in value]):
            raise ValueError("At least one column must have 'id' as a key.")
        self._columns_keys = value

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get the list of tasks from the state JSON."""
        tasks = DataJson()[self.widget_id]["tasks"]
        return tasks

    def update_task(self, task_id: int, task: Dict[str, Any]):
        tasks = self.get_tasks()
        for task in tasks:
            if task["id"] == task_id:
                task.update(task)
                DataJson()[self.widget_id]["tasks"] = tasks
                DataJson().send_changes()
                return
        raise KeyError(f"Task with id {task_id} not found in tasks list.")

    def add_task(self, task: Dict[str, Any]):
        """Add a task to the tasks list in the state JSON."""
        if not isinstance(task, dict):
            raise TypeError("task must be a dictionary")
        tasks = self.get_tasks()
        tasks.append(task)
        DataJson()[self.widget_id]["tasks"] = tasks
        DataJson().send_changes()

    def _get_table_data(self) -> List[List[Any]]:
        tasks = self.get_tasks()
        data = []
        for task in tasks:
            row = [self._get_task_item(col, task, default="unknown") for col in self.columns_keys]
            data.append(row)
        return data

    @property
    def logs(self):
        if not hasattr(self, "_logs"):
            self._logs = TaskLogs()
        return self._logs

    @property
    def logs_modal(self):
        if not hasattr(self, "_logs_modal"):
            self._logs_modal = Dialog(title="Task logs", content=self.logs)
        return self._logs_modal

    def _create_tasks_history_table(self):
        columns = self.table_columns
        return FastTable(columns=columns, sort_column_idx=0, fixed_columns=1, sort_order="desc")

    @property
    def table(self):
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_history_table()

            @self._tasks_table.row_click
            def on_row_click(clicked_row: FastTable.ClickedRow):
                self.logs.set_task_id(clicked_row.row[0])
                self.logs_modal.show()

        return self._tasks_table

    def update(self):
        for task in self.get_tasks():
            task_id = task["id"]
            task_info = self.api.task.get_info_by_id(task_id)
            task.update(task_info)
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

    def _autorefresh(self):
        t = time.monotonic()
        while not self._stop_autorefresh:
            if time.monotonic() - t >= self._refresh_interval:
                t = time.monotonic()
                try:
                    self.update()
                except Exception as e:
                    logger.debug(f"Error during autorefresh: {e}")
            time.sleep(1)

    def stop_autorefresh(self, wait: bool = False):
        self._stop_autorefresh = True
        if wait:
            if self._refresh_thread is not None:
                self._refresh_thread.join()

    def start_autorefresh(self, interval: int = 60):
        self._refresh_interval = interval
        self._stop_autorefresh = False
        if self._refresh_thread is None:
            self._refresh_thread = threading.Thread(target=self._autorefresh, daemon=True)
        if not self._refresh_thread.is_alive():
            self._refresh_thread.start()

    def get_json_data(self):
        return {"tasks": []}

    def get_json_state(self):
        return {}

    def to_html(self):
        return self.table.to_html()
