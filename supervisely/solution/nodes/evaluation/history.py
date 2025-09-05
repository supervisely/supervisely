from typing import Any, Dict

from supervisely import logger
from supervisely.app.widgets import FastTable, TasksHistory, Button


class EvaluationTaskHistory(TasksHistory):
    def __init__(
        self,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id)
        self._table_columns = [
            "Task ID",
            "Model Path",
            "Status",
            "Collection Name",
            "Session ID",
        ]
        self._columns_keys = [
            ["taskId"],
            ["modelPath"],
            ["status"],
            ["collectionName"],
            ["sessionId"],
        ]

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Dict[str, Any]) -> int:
        super().add_task(task)
        self.update()

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
                logger.debug("Showing logs for task ID: %s", self.logs.get_task_id())
                self.logs_modal.show()

        return self._tasks_table

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
