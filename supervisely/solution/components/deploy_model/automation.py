from typing import Callable, Optional

from supervisely.solution.components import Automation


class DeployTasksAutomation(Automation):
    REFRESH_RATE = 30  # seconds
    REFRESH_GPU_USAGE = "refresh_gpu_usage"

    def apply(self, func: Optional[Callable], job_id: str, sec: int = None) -> None:
        if self.scheduler.is_job_scheduled(job_id):
            self.scheduler.remove_job(job_id)
        sec = sec or self.REFRESH_RATE
        self.scheduler.add_job(func, sec, job_id)

    def remove(self, job_id: str) -> None:
        if self.scheduler.is_job_scheduled(job_id):
            self.scheduler.remove_job(job_id)
