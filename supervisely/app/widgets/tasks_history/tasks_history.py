import threading
import time
from typing import Any, Dict, List, Union
from venv import logger

from supervisely.api.api import Api
from supervisely.app import DataJson
from supervisely.app.widgets import Button
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.app.widgets.grid_gallery_v2.grid_gallery_v2 import GridGalleryV2
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

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self):
        return {"tasks": []}

    def get_json_state(self):
        return {}

    # ------------------------------------------------------------------
    # Table ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _on_table_row_click(self, clicked_row: FastTable.ClickedRow):
        self.logs.set_task_id(clicked_row.row[0])
        self.logs_modal.show()

    @property
    def table(self):
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_history_table()

            @self._tasks_table.row_click
            def on_row_click(clicked_row: FastTable.ClickedRow):
                self._on_table_row_click(clicked_row)

        return self._tasks_table

    def to_html(self):
        return self.table.to_html()

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

    def _get_table_data(self) -> List[List[Any]]:
        tasks = self.get_tasks()
        data = []
        for task in tasks:
            row = [self._get_task_item(col, task, default="unknown") for col in self.columns_keys]
            data.append(row)
        return data

    def _create_tasks_history_table(self):
        columns = self.table_columns
        return FastTable(columns=columns, sort_column_idx=0, fixed_columns=1, sort_order="desc")

    @table_columns.setter
    def table_columns(self, value: List[str]):
        self._table_columns = value

    def update(self):
        for task in self.get_tasks():
            try:
                task_id = task["id"]
            except KeyError:
                task_id = task.get("task_info", {}).get("id", None)
                if task_id is None:
                    continue
            task_info = self.api.task.get_info_by_id(task_id)
            for col_keys in self.columns_keys:
                if not isinstance(col_keys, list):
                    col_keys = [col_keys]
                task_item = self._get_task_item(col_keys, task_info, default=None)
                if task_item is not None:
                    task[col_keys[-1]] = task_item
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

    # ------------------------------------------------------------------
    # Table Helpers ----------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tasks ------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get the list of tasks from the state JSON."""
        tasks = DataJson()[self.widget_id]["tasks"]
        return tasks

    def add_task(self, task: Dict[str, Any]):
        """Add a task to the tasks list in the state JSON."""
        if not isinstance(task, dict):
            raise TypeError("task must be a dictionary")
        tasks = self.get_tasks()
        task = self._prepare_task_item(task)
        tasks.append(task)
        DataJson()[self.widget_id]["tasks"] = tasks
        DataJson().send_changes()

    def update_task(self, task_id: int, task: Dict[str, Any]):
        tasks = self.get_tasks()
        for row in tasks:
            if row["id"] == task_id:
                row.update(task)
                DataJson()[self.widget_id]["tasks"] = tasks
                DataJson().send_changes()
                return
        raise KeyError(f"Task with id {task_id} not found in tasks list.")

    def _get_task_item(
        self,
        keys: Union[str, int, List[Union[str, int]]],
        task: Dict[str, Any],
        default=_MISSING,
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

    def _prepare_task_item(self, task: Dict[str, Any]) -> List[Any]:
        """Prepares a task item for saving to DataJson."""
        res = {}
        for col_keys in self.columns_keys:
            if not isinstance(col_keys, list):
                col_keys = [col_keys]
            if len(col_keys) == 1:
                res[col_keys[0]] = self._get_task_item(col_keys, task, default="unknown")
            else:
                # If col_keys is a list, set value in a nested dictionary
                current = res
                for key in col_keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[col_keys[-1]] = self._get_task_item(col_keys, task, default="unknown")
        return res

    # ------------------------------------------------------------------
    # Logs --------------------------------------------------------------
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Gallery ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def gallery(self) -> GridGalleryV2:
        if not hasattr(self, "_gallery"):
            self._gallery = GridGalleryV2(
                columns_number=3,
                enable_zoom=False,
                enable_pagination=True,
                pagination_page_size=9,
            )
        return self._gallery

    @property
    def preview_modal(self) -> Dialog:
        if not hasattr(self, "_preview_modal"):
            self._preview_modal = Dialog(
                title="Preview",
                content=self.gallery,
                size="small",
            )
        return self._preview_modal

    # ------------------------------------------------------------------
    # Modal ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Dialog window that displays the tasks table."""
        if not hasattr(self, "_modal") or self._modal is None:
            self._modal = Dialog(title="Tasks History", content=self.table)
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Small button that opens the history modal."""
        if not hasattr(self, "_open_modal_button") or self._open_modal_button is None:
            btn = Button(
                text="Tasks History",
                icon="zmdi zmdi-format-list-bulleted",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click
            def _on_click():
                self.update()
                self.modal.show()

            self._open_modal_button = btn
        return self._open_modal_button
