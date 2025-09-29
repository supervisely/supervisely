import datetime
from typing import Any, Dict, List, Union

from supervisely import logger
from supervisely.app.content import DataJson
from supervisely.solution.components import TasksHistoryWidget


class SendEmailHistory(TasksHistoryWidget):

    class Item:
        class Status:
            SENT = "Sent"
            FAILED = "Failed"
            PENDING = "Pending"

        def __init__(
            self,
            sent_to: Union[List, str],
            status: str = None,
            created_at: str = None,
        ):
            """
            Initialize a notification with the recipient, origin, status, and creation time.
            """
            self.sent_to = sent_to
            self.status = status or self.Status.PENDING
            self.created_at = created_at or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def to_json(self) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
            """
            Convert the notification history to a JSON serializable format.
            """
            res = {
                "sent_to": self.sent_to if isinstance(self.sent_to, list) else [self.sent_to],
            }
            if self.status:
                res["status"] = self.status
            if self.created_at:
                res["created_at"] = self.created_at
            return res

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "ID",
                "Created At",
                "Sent To",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["id"],
                ["created_at"],
                ["sent_to"],
                ["status"],
            ]
        return self._columns_keys

    def update(self):
        self.table.clear()
        for task in self._get_table_data():
            self.table.insert_row(task)

    def add_task(self, task: Union["SendEmailHistory.Item", Dict[str, Any]]):
        if isinstance(task, SendEmailHistory.Item):
            task = task.to_json()
        task["id"] = len(self.get_tasks()) + 1  # Assign a new ID
        super().add_task(task)
        self.update()
        return task["created_at"]

    def update_task(
        self,
        time: datetime.datetime,
        task: Union["SendEmailHistory.Item", Dict[str, Any]],
    ):
        if isinstance(task, SendEmailHistory.Item):
            task = task.to_json()
        tasks = self.get_tasks()
        for row in tasks:
            if row["created_at"] == time:
                row.update(task)
                DataJson()[self.widget_id]["tasks"] = tasks
                DataJson().send_changes()
                return
        raise KeyError(f"Task with created_at {time} not found in the notification history.")

    @property
    def table(self):
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_history_table()
        return self._tasks_table
