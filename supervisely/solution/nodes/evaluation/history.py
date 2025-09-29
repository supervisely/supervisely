from typing import Any, Dict, List

from supervisely import logger
from supervisely.app.widgets import Button, FastTable, TasksHistory
from supervisely.solution.components import TasksHistoryWidget
from supervisely.solution.engine.modal_registry import ModalRegistry


class EvaluationTaskHistory(TasksHistoryWidget):

    @property
    def table(self):
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_history_table()

            @self._tasks_table.cell_click
            def on_cell_click(clicked_cell: FastTable.ClickedCell):
                if clicked_cell.column_index == 4:  # Session ID
                    col_idx = 4
                else:
                    col_idx = 0
                self.logs.set_task_id(clicked_cell.row[col_idx])
                ModalRegistry().open_logs(owner_id=self.widget_id)

        return self._tasks_table

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Model Path",
                "Status",
                "Collection Name",
                "Session ID",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["taskId"],
                ["modelPath"],
                ["status"],
                ["collectionName"],
                ["sessionId"],
            ]
        return self._columns_keys

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Dict[str, Any]) -> int:
        super().add_task(task)
        self.update()

    @property
    def btn(self) -> Button:
        if not hasattr(self, "_task_history_btn"):
            self._task_history_btn = Button(
                "Tasks History",
                icon="zmdi zmdi-format-list-bulleted",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @self._task_history_btn.click
            def show_task_history():
                self.modal.show()

        return self._task_history_btn
