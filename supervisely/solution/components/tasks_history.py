from typing import Any, Dict, List

from supervisely.api.api import Api
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory
from supervisely.solution.components.base.node import SolutionElement
from supervisely.solution.scheduler import TasksScheduler


class SolutionTasksHistory(SolutionElement, TasksScheduler):
    def __init__(self, api: Api, title: str = "Tasks History"):
        self.api = api
        self.tasks_history = TasksHistory()
        self.tasks_modal = Dialog(title=title, content=self.tasks_history)
        self.logs_modal = self.tasks_history.logs_modal

    def add_task(self, task: Dict[str, Any]):
        self.tasks_history.add_task(task)

    @property
    def modals(self) -> List[Dialog]:
        return [self.tasks_modal, self.logs_modal]
