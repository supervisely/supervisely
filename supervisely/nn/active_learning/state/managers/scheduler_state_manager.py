from datetime import datetime
from typing import Any, Dict, Optional


class BackgroundTask:
    """Represents a scheduled background task with metadata"""

    def __init__(
        self,
        job_id: str,
        interval: int,
        enabled: bool = True,
        last_run: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.interval = interval  # in seconds
        self.enabled = enabled
        self.last_run = last_run  # ISO format datetime string
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage"""
        return {
            "job_id": self.job_id,
            "interval": self.interval,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackgroundTask":
        """Create task from dictionary"""
        return cls(
            job_id=data["job_id"],
            interval=data["interval"],
            enabled=data["enabled"],
            last_run=data["last_run"],
            metadata=data.get("metadata", {}),
        )


class SchedulerStateManager:
    """
    Manages state of background tasks for active learning workflow.
    Allows persisting and restoring task schedules across application restarts.
    """

    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.tasks_key = "background_tasks"

    def get_all_tasks(self) -> Dict[str, BackgroundTask]:
        """Get all registered background tasks"""
        tasks_data = self.state_manager.get(self.tasks_key, {})
        return {
            task_id: BackgroundTask.from_dict(task_data)
            for task_id, task_data in tasks_data.items()
        }

    def get_task(self, job_id: str) -> Optional[BackgroundTask]:
        """Get a specific background task by job_id"""
        tasks = self.get_all_tasks()
        return tasks.get(job_id)

    def add_task(
        self, job_id: str, interval: int, metadata: Optional[Dict[str, Any]] = None
    ) -> BackgroundTask:
        """
        Add or update a background task

        Args:
            job_id: Unique identifier for the task
            interval: Task interval in seconds
            metadata: Additional task metadata

        Returns:
            BackgroundTask: The created/updated task
        """
        tasks_data = self.state_manager.get(self.tasks_key, {})

        # Create or update task
        task = BackgroundTask(
            job_id=job_id, interval=interval, enabled=True, last_run=None, metadata=metadata or {}
        )

        # If task exists, preserve some fields
        if job_id in tasks_data:
            existing = tasks_data[job_id]
            task.enabled = True  # existing.get("enabled", True)  # ! TODO: Check if this is correct
            task.last_run = existing.get("last_run")
            task.metadata = metadata
            # Merge metadata
            # if existing.get("metadata"):
            #     merged_metadata = existing["metadata"].copy()
            #     if metadata:
            #         merged_metadata.update(metadata)
            #         task.metadata = merged_metadata

        # Save task
        tasks_data[job_id] = task.to_dict()
        self.state_manager.set(self.tasks_key, tasks_data)

        return task

    def update_task(self, job_id: str, **kwargs) -> Optional[BackgroundTask]:
        """
        Update properties of an existing task

        Args:
            job_id: Task identifier
            **kwargs: Properties to update (interval, enabled, metadata)

        Returns:
            BackgroundTask or None: Updated task or None if task not found
        """
        tasks_data = self.state_manager.get(self.tasks_key, {})

        if job_id not in tasks_data:
            return None

        # Update task properties
        task_data = tasks_data[job_id]

        if "interval" in kwargs:
            task_data["interval"] = kwargs["interval"]

        if "enabled" in kwargs:
            task_data["enabled"] = kwargs["enabled"]

        if "metadata" in kwargs:
            task_data["metadata"] = task_data.get("metadata", {})
            task_data["metadata"].update(kwargs["metadata"])

        # Save changes
        tasks_data[job_id] = task_data
        self.state_manager.set(self.tasks_key, tasks_data)

        return BackgroundTask.from_dict(task_data)

    def record_task_run(self, job_id: str) -> bool:
        """
        Record that a task has been executed

        Args:
            job_id: Task identifier

        Returns:
            bool: True if task exists and was updated, False otherwise
        """
        tasks_data = self.state_manager.get(self.tasks_key, {})

        if job_id not in tasks_data:
            return False

        # Update last run timestamp
        tasks_data[job_id]["last_run"] = datetime.now().isoformat()
        self.state_manager.set(self.tasks_key, tasks_data)

        return True

    def remove_task(self, job_id: str) -> bool:
        """
        Remove a background task

        Args:
            job_id: Task identifier

        Returns:
            bool: True if task was removed, False if it didn't exist
        """
        tasks_data = self.state_manager.get(self.tasks_key, {})

        if job_id not in tasks_data:
            return False

        # Remove task
        del tasks_data[job_id]
        self.state_manager.set(self.tasks_key, tasks_data)

        return True

    def enable_task(self, job_id: str, enabled: bool = True) -> bool:
        """
        Enable or disable a background task

        Args:
            job_id: Task identifier
            enabled: Whether task should be enabled

        Returns:
            bool: True if task exists and was updated, False otherwise
        """
        return self.update_task(job_id, enabled=enabled) is not None

    def get_enabled_tasks(self) -> Dict[str, BackgroundTask]:
        """Get all enabled tasks"""
        return {task_id: task for task_id, task in self.get_all_tasks().items() if task.enabled}

    # def get_disabled_tasks(self) -> Dict[str, BackgroundTask]:
    #     """Get all disabled tasks"""
    #     return {task_id: task for task_id, task in self.get_all_tasks().items() if not task.enabled}
