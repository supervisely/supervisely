from typing import Dict, Union
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely import is_development
from supervisely.app.widgets import Button, Flexbox, InputNumber, Widget


class TaskLogs(Widget):
    def __init__(
        self,
        task_id: int,
        multiple: bool = False,
        filterable: bool = True,
        widget_id: str = None,
    ):
        self._task_id = task_id
        self._is_development = is_development()
        self._multiple = multiple
        self._filterable = filterable

        self._task_id_input = InputNumber(value=self._task_id, size="small", controls=False)
        self._task_logs_stop = Button("Change task id", button_size="mini", plain=True)
        self._task_id_change_btn = Button("OK", button_size="mini", plain=True)
        self._task_id_input.hide()
        self._task_id_change_btn.hide()
        self._task_id_change_controls = Flexbox(
            widgets=[self._task_id_input, self._task_logs_stop, self._task_id_change_btn],
        )

        @self._task_logs_stop.click
        def stop_logs_ws():
            self._task_logs_stop.loading = True
            self._set_task_id(None)
            self._task_logs_stop.loading = False
            self._task_logs_stop.hide()
            self._task_id_input.show()
            self._task_id_change_btn.show()

        @self._task_id_change_btn.click
        def change_task_id():
            self._task_id_change_btn.loading = True
            new_task_id = int(self._task_id_input.value)
            self._set_task_id(new_task_id)
            self._task_id_change_btn.loading = False
            self._task_id_change_btn.hide()
            self._task_id_input.hide()
            self._task_logs_stop.show()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "taskId": self._task_id,
            "options": {"multiple": self._multiple, "filterable": self._filterable},
        }

    def get_json_state(self) -> Dict:
        return {}

    def get_task_id(self) -> int:
        return DataJson()[self.widget_id]["taskId"]

    def _set_task_id(self, task_id: Union[int, None]):
        self._task_id = task_id
        DataJson()[self.widget_id]["taskId"] = self._task_id
        DataJson().send_changes()
