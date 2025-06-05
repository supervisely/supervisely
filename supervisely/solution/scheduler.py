class TasksScheduler:
    """
    This class is used to schedule tasks using APScheduler.
    It provides methods to add, remove, and modify scheduled tasks.
    """

    def __init__(self):
        try:
            from apscheduler.jobstores.memory import MemoryJobStore
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            raise ImportError(
                "Please install 'apscheduler' package to use the TasksScheduler class. "
                "You can do this by running 'pip install apscheduler'."
            )

        self.scheduler = BackgroundScheduler(jobstores={"default": MemoryJobStore()})
        self.jobs = {}
        self.scheduler.start()

    def add_job(self, job_id, func, sec, replace_existing=True, args=None):
        """Add a new scheduled job"""
        job = self.scheduler.add_job(
            func, "interval", args=args, seconds=sec, id=job_id, replace_existing=replace_existing
        )
        self.jobs[job_id] = job
        return job

    def remove_job(self, job_id):
        """Remove a scheduled job"""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            return True
        return False

    def modify_interval(self, job_id, interval_seconds):
        """Modify the interval of an existing job"""
        if job_id in self.jobs:
            self.scheduler.reschedule_job(job_id, trigger="interval", seconds=interval_seconds)
            return True
        return False

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()

    def is_job_scheduled(self, job_id) -> bool:
        """Check if a job is scheduled"""
        return job_id in self.jobs
