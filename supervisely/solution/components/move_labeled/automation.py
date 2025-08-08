from typing import Callable

from supervisely.solution.base_node import Automation


class MoveLabeledAuto(Automation):

    def __init__(self):
        super().__init__()
        self.job_id = "move_labeled_automation_job"
        self.func = None

    def apply(self, func: Callable[[], None], sec: int, *args) -> None:
        self.func = func
        if sec is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(
                self.func, interval=sec, job_id=self.job_id, replace_existing=True, *args
            )
