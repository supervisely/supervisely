from copy import deepcopy
from typing import Any, Dict, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.sly_logger import logger


class ProjectStateManager:
    """Manages project-related operations"""

    def __init__(self, api: Api, project: Union[int, ProjectInfo]):
        self.api = api
        if isinstance(project, int):
            self.project_id = project
            self.project = api.project.get_info_by_id(project)
        else:
            self.project_id = project.id
            self.project = project

    def refresh(self) -> None:
        """Refresh project info from server"""
        self.project = self.api.project.get_info_by_id(self.project_id)

    def get_project(self) -> ProjectInfo:
        """Get current project info"""
        return self.project

    def create_related_project(self, suffix: str) -> ProjectInfo:
        """Create a related project with the given suffix"""
        workspace_id = self.project.workspace_id
        name = f"{self.project.name} ({suffix})"
        new_project = self.api.project.create(workspace_id, name, change_name_if_conflict=True)
        logger.info(f"Created {suffix} project: {new_project.name} ({new_project.id})")
        return new_project


class StateManager:
    """Manages state persistence and retrieval"""

    def __init__(
        self, api: Api, project_manager: ProjectStateManager, solution_key: str = "solutions"
    ):
        self.api = api
        self.project_manager = project_manager
        self.solution_key = solution_key
        self.state: Dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        """Load state from server"""
        self.project_manager.refresh()  # Ensure project info is up-to-date
        project = self.project_manager.get_project()
        if project.custom_data and self.solution_key in project.custom_data:
            self.state = deepcopy(project.custom_data.get(self.solution_key, {}))
        else:
            self.state = {}

    def get(self, key: str, default: Any = None, force_refresh: bool = True) -> Any:
        """Get value for key with optional default"""
        if force_refresh:
            self.refresh()  # Ensure state is up-to-date
        return self.state.get(key, default)

    def set(self, key: str, value: Any, force_refresh: bool = True) -> None:
        """Set value for key and persist changes"""
        if force_refresh:
            self.refresh()  # Ensure state is up-to-date
        self.state[key] = value
        self._persist()

    def update(self, key: str, value: Any, force_refresh: bool = True) -> None:
        """Update multiple keys at once and persist"""
        if force_refresh:
            self.refresh()  # Ensure state is up-to-date
        self.state.update({key: value})
        self._persist()

    def _persist(self) -> None:
        """Save state to server"""
        project = self.project_manager.get_project()
        custom_data = project.custom_data or {}
        custom_data[self.solution_key] = deepcopy(self.state)
        self.api.project.update_custom_data(project.id, custom_data)
        self.refresh()  # Reload to ensure consistency
