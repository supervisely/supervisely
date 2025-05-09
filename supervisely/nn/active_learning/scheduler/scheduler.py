import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from supervisely.nn.active_learning.state.state import ALState


class SchedulerJobs:
    LABELING_QUEUE_STATS = "check_labeling_info"
    CLOUD_IMPORT = "import_from_cloud"
    RUN_DATA_ORGANIZER = "run_data_organizer"
    MOVE_TO_TRAINING = "move_to_training_project"
    REFRESH_PROJECT_INFO = "refresh_project_info"
    START_SAMPLING = "start_sampling"
    SEND_SAMPLING_REQUEST = "send_sampling_request"
    START_RANDOM_SAMPLING = "start_random_sampling"


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
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        interval = "Every"
        if h > 0:
            interval += f" {h}h"
        if m > 0:
            interval += f" {m}m"
        if s > 0:
            interval += f" {s}s"
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


class PersistentTasksScheduler:
    """
    Extends TasksScheduler to provide persistence of task scheduling across application restarts.
    Uses ALState's BackgroundTasksStateManager for persistence.
    """

    def __init__(self, al_state: ALState):
        """
        Initialize the persistent scheduler

        Args:
            al_state: Active Learning state manager
        """
        self.al_state = al_state
        self.scheduler = TasksScheduler()

    def _wrap_function(self, func: Callable, job_id: str) -> Callable:
        """
        Wrap a function to record execution in state manager

        Args:
            func: Original function to wrap
            job_id: Task identifier

        Returns:
            Callable: Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.al_state.record_task_execution(job_id)
            return result

        return wrapper

    def add_job(
        self,
        job_id: str,
        func: Callable,
        sec: int,
        replace_existing: bool = True,
        args: Optional[Tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a new scheduled job with persistence

        Args:
            job_id: Unique task identifier
            func: Function to execute
            sec: Interval in seconds
            replace_existing: Whether to replace existing job with same ID
            args: Arguments to pass to the function
            metadata: Optional metadata for the task

        Returns:
            Job: The scheduled job
        """
        # Register task in state manager
        self.al_state.register_background_task(job_id, sec, metadata)

        # Schedule the job with a wrapped function that records execution
        wrapped_func = self._wrap_function(func, job_id)
        return self.scheduler.add_job(job_id, wrapped_func, sec, replace_existing, args)

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job

        Args:
            job_id: Task identifier

        Returns:
            bool: True if job was removed
        """
        # Remove from scheduler
        scheduler_result = self.scheduler.remove_job(job_id)

        # Disable in state manager but keep the record
        self.al_state.enable_background_task(job_id, False)

        return scheduler_result

    def modify_interval(self, job_id: str, interval_seconds: int) -> bool:
        """
        Modify the interval of an existing job

        Args:
            job_id: Task identifier
            interval_seconds: New interval in seconds

        Returns:
            bool: True if job was modified
        """
        # Update in scheduler
        scheduler_result = self.scheduler.modify_interval(job_id, interval_seconds)

        # Update in state manager
        self.al_state.update_background_task(job_id, interval=interval_seconds)

        return scheduler_result

    def restore_jobs(self, job_functions: Dict[str, Callable]) -> List[str]:
        """
        Restore jobs from state manager

        Args:
            job_functions: Dictionary mapping job IDs to their functions

        Returns:
            List[str]: List of restored job IDs
        """
        restored_jobs = []
        enabled_tasks = self.al_state.get_enabled_background_tasks()

        for job_id, task_data in enabled_tasks.items():
            if job_id in job_functions:
                func = job_functions[job_id]
                interval = task_data["interval"]
                metadata = task_data.get("metadata", {})
                args = metadata.get("args")

                # Schedule the job
                wrapped_func = self._wrap_function(func, job_id)
                self.scheduler.add_job(job_id, wrapped_func, interval, True, args)
                restored_jobs.append(job_id)

        return restored_jobs

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
