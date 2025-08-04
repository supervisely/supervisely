from typing import Callable, Optional
from supervisely.sly_logger import logger
from supervisely.solution.base_node import Automation


class ProjectAutomation(Automation):
    """
    Automation for refreshing labeling queue information periodically
    """

    def __init__(self, project_id: int, func: Optional[Callable[[], None]] = None):
        super().__init__()
        self.job_id = f"refresh_project_{project_id}"
        self.project_id = project_id
        self.func = func

    def apply(self, sec: int, *args) -> None:
        self.scheduler.add_job(
            self.func, interval=sec, job_id=self.job_id, replace_existing=True, *args
        )

    def schedule_refresh(self, func: Callable[[], None], interval_sec: int = 5) -> None:
        """
        Schedule a job to refresh labeling queue info.
        """
        self.scheduler.add_job(
            func, interval=interval_sec, job_id=self.job_id, replace_existing=True
        )
        logger.info(
            f"Scheduled refresh for labeling queue {self.project_id} every {interval_sec} seconds"
        )

    def unschedule_refresh(self) -> None:
        """
        Unschedule the job that refreshes labeling queue info.
        """
        if self.scheduler.is_job_scheduled(self.job_id):
            self.scheduler.remove_job(self.job_id)
            logger.info(f"Unscheduled refresh for labeling queue {self.project_id}")
