from typing import Dict, List, Optional

from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)


class TrainingStateManager:
    """Handles training tasks"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    @property
    def project_id(self) -> Optional[int]:
        """Get training project ID"""
        return self.state_manager.get("training_project_id")

    def set_training_project_id(self, project_id: int) -> None:
        """Set training project ID"""
        self.state_manager.set("training_project_id", project_id)

    def get_training_tasks(self) -> Dict[str, List[int]]:
        """Get all training tasks"""
        return self.state_manager.get("training_tasks", {})

    def add_training_tasks(self, slug: str, task_ids: List[int]) -> None:
        """Add training task IDs to state"""
        tasks = self.get_training_tasks()

        if slug not in tasks:
            tasks[slug] = []

        for task_id in task_ids:
            if task_id not in tasks[slug]:
                tasks[slug].append(task_id)

        self.state_manager.set("training_tasks", tasks)
