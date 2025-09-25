from supervisely.api.api import Api
from supervisely.app.widgets import Button, TasksHistory


class TrainingDataHistory(TasksHistory):
    def __init__(self, api: Api):
        super().__init__(api)
        self._table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Started At",
            "Images Count",
            "Status",
        ]
        self.columns_keys = [
            ["task_id"],
            ["app", "name"],
            ["datasets", "id"],
            ["timestamp"],
            ["items_count"],
            ["status"],
        ]

    def update(self):
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

    def add_task(self, task):
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
