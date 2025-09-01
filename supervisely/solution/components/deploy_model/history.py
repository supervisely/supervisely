from supervisely.api.api import Api
from supervisely.app.widgets import Button, TasksHistory


class DeployTasksHistory(TasksHistory):
    def __init__(self, api: Api):
        super().__init__(api)
        self.table_columns = [
            "Task ID",
            "App Name",
            "Model Name",
            "Started At",
            # "Classses Count",
            "Runtime",
            "Hardware",
            "Device",
        ]
        self.columns_keys = [
            ["id"],
            ["app_name"],
            ["model_name"],
            ["started_at"],
            # ["meta", "model", "classes_count"],
            ["runtime"],
            ["hardware"],
            ["device"],
        ]

    def update(self):
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

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
