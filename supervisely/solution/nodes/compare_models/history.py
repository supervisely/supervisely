from typing import Any, Dict

from supervisely.app.widgets import Button, TasksHistory


class ComparisonHistory(TasksHistory):

    def __init__(
        self,
        widget_id: str = None,
    ):
        super().__init__(widget_id=widget_id)
        self._table_columns = [
            "Task ID",
            "Created At",
            # "Input Evaluations",
            "Comparison Report",
            "Best checkpoint",
        ]
        self._columns_keys = [
            ["id"],
            ["created_at"],
            # ["evaluation_dirs"],
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

    @property
    def history_btn(self) -> Button:
        if not hasattr(self, "_history_btn"):
            self._history_btn = Button(
                "History",
                icon="zmdi zmdi-format-list-bulleted",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @self._history_btn.click
            def show_history():
                self.modal.show()

        return self._history_btn
