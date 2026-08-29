from typing import Dict, List

from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)


class ImportStateManager:
    """Handles import tasks"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def get_import_tasks(self) -> Dict[str, List[int]]:
        """Get all import tasks"""
        return self.state_manager.get("import_tasks", {})

    def add_import_tasks(self, slug: str, task_ids: List[int]) -> None:
        """Add import task IDs to state"""
        tasks = self.get_import_tasks()

        if slug not in tasks:
            tasks[slug] = []

        for task_id in task_ids:
            if task_id not in tasks[slug]:
                tasks[slug].append(task_id)

        self.state_manager.set("import_tasks", tasks)
