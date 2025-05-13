import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.nn.active_learning.state.managers import (
    BackgroundTask,
    ImportStateManager,
    LabelingStateManager,
    ProjectStateManager,
    SamplingStateManager,
    SchedulerStateManager,
    StateManager,
    TrainingStateManager,
    TrainValSplitStateManager,
)
from supervisely.sly_logger import logger


class ALState:
    """
    Class to manage Active Learning state for Supervisely Solutions.
    Uses a modular design with specialized manager classes.

    Handles:
    - Import tasks
    - Sampling tasks
    - Labeling queue tasks
    - Train/Val split settings
    - Sampling settings
    - Training tasks
    """

    def __init__(
        self,
        api: Api,
        project: Union[int, ProjectInfo],
        solution_key: str = "solutions",
    ):
        """
        Initialize the Active Learning state manager

        Args:
            api: Supervisely API instance
            project: Project ID or ProjectInfo object
            solution_key: Key under which solution data is stored in custom_data
        """
        # Core components
        self.project_manager = ProjectStateManager(api, project)
        self.state_manager = StateManager(api, self.project_manager, solution_key)

        # Specialized managers
        self.import_manager = ImportStateManager(self.state_manager)
        self.sampling_manager = SamplingStateManager(self.state_manager)
        self.labeling_manager = LabelingStateManager(self.state_manager, api)
        self.split_manager = TrainValSplitStateManager(self.state_manager)
        self.training_manager = TrainingStateManager(self.state_manager)

        # Background tasks state
        self.background_tasks = SchedulerStateManager(self.state_manager)

        # Initialize resources if needed
        self._initialize_resources()

    # create from workspace and team
    @classmethod
    def create_from_workspace(cls, api: Api, workspace_id: int, project_name: str) -> "ALState":
        """
        Create ALState from workspace

        Args:
            api: Supervisely API instance
            workspace_id: Workspace ID

        Returns:
            ALState instance
        """
        project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
        time.sleep(0.1)  # Give some time for the project to be created
        logger.info(f"Created project: {project.name} ({project.id})")
        return cls(api, project)

    @classmethod
    def create_from_team(cls, api: Api, team_id: int, project_name: str) -> "ALState":
        """
        Create ALState from team

        Args:
            api: Supervisely API instance
            team_id: Team ID

        Returns:
            ALState instance
        """
        ws = api.workspace.create(team_id, "Solutions Workspace", change_name_if_conflict=True)
        time.sleep(0.1)  # Give some time for the workspace to be created
        logger.info(f"Created workspace: {ws.name} ({ws.id})")
        project = api.project.create(ws.id, project_name, change_name_if_conflict=True)
        time.sleep(0.1)  # Give some time for the project to be created
        logger.info(f"Created project: {project.name} ({project.id})")
        return cls(api, project)

    def _initialize_resources(self) -> None:
        """Initialize required resources if they don't exist"""
        api = self.state_manager.api
        project_manager = self.project_manager

        # Create labeling project if needed
        if not self.labeling_manager.project_id:
            new_project = project_manager.create_related_project("labeling")
            self.labeling_manager.set_labeling_project_id(new_project.id)

        # Create training project if needed
        if not self.training_manager.project_id:
            new_project = project_manager.create_related_project("training")
            self.training_manager.set_training_project_id(new_project.id)

        # Create collection if needed
        labeling_project_id = self.labeling_manager.project_id
        if labeling_project_id and not self.labeling_manager.collection_id:
            name = "Collection for Solutions"
            new_collection = api.entities_collection.create(labeling_project_id, name)
            self.labeling_manager.set_labeling_collection_id(new_collection.id)
            logger.info(f"Created entities collection: {new_collection.name} ({new_collection.id})")

        # Create labeling queue if needed
        collection_id = self.labeling_manager.collection_id
        if collection_id and not self.labeling_manager.queue_id:
            name = "Collection for Solutions"
            user_ids = [api.user.get_my_info().id]
            new_queue = api.labeling_queue.create(
                name=name,
                user_ids=user_ids,
                reviewer_ids=user_ids,
                collection_id=collection_id,
                dynamic_classes=True,
                dynamic_tags=True,
                allow_review_own_annotations=True,
                skip_complete_job_on_empty=True,
            )
            self.labeling_manager.set_labeling_queue_id(new_queue)
            logger.info(f"Created labeling queue ID:{new_queue}")

    def refresh(self) -> None:
        """Refresh state from server"""
        self.state_manager.refresh()

    # Project-related methods
    @property
    def project_id(self) -> int:
        return self.project_manager.project_id

    @property
    def project(self) -> ProjectInfo:
        return self.project_manager.project

    @property
    def labeling_project_id(self) -> int:
        return self.labeling_manager.project_id

    @property
    def training_project_id(self) -> int:
        return self.training_manager.project_id

    @property
    def labeling_queue_id(self) -> int:
        return self.labeling_manager.queue_id

    @labeling_queue_id.setter
    def labeling_queue_id(self, queue_id: int) -> None:
        self.labeling_manager.set_labeling_queue_id(queue_id)

    @property
    def labeling_collection_id(self) -> int:
        return self.labeling_manager.collection_id

    # Import-related methods
    @property
    def import_tasks(self) -> Dict[str, List[int]]:
        return self.import_manager.get_import_tasks()

    def add_import_task(self, slug: str, task_id: int) -> None:
        self.import_manager.add_import_tasks(slug, [task_id])

    def add_import_tasks(self, slug: str, task_ids: List[int]) -> None:
        self.import_manager.add_import_tasks(slug, task_ids)

    # Sampling-related methods
    def save_sampling_settings(self, settings: dict) -> None:
        self.sampling_manager.save_sampling_settings(settings=settings)

    def get_sampling_settings(self) -> Dict[str, Any]:
        return self.sampling_manager.get_sampling_settings()

    def get_sampling_mode(self) -> str:
        return self.sampling_manager.get_sampling_mode()

    def get_sampling_tasks(self) -> List[int]:
        return self.sampling_manager.get_sampling_tasks()

    def add_sampling_task(self, task_id: int) -> None:
        self.sampling_manager.add_sampling_tasks([task_id])

    def add_sampling_tasks(self, task_ids: List[int]) -> None:
        self.sampling_manager.add_sampling_tasks(task_ids)

    def add_sampling_batch(self, batch_data: Dict[int, List[int]]) -> None:
        self.sampling_manager.add_sampling_batch(batch_data)

    def get_sampled_images(self) -> Dict[str, List[int]]:
        return self.sampling_manager.get_sampled_images()

    # Labeling-related methods
    def get_new_labeled_images(self) -> List[int]:
        return self.labeling_manager.get_new_labeled_images()

    # Split-related methods
    def add_split_collection(self, key: str, collection_id: int) -> None:
        self.split_manager.add_split_collections(key, [collection_id])

    def add_split_collections(self, key: str, collection_ids: List[int]) -> None:
        self.split_manager.add_split_collections(key, collection_ids)

    def get_split_collection(self, key: str) -> List[int]:
        return self.split_manager.get_split_collections().get(key, [])

    def set_split_settings(
        self,
        mode: Literal["random", "datasets"],
        train: Union[float, List[str]],
        val: Union[float, List[str]],
    ) -> None:
        self.split_manager.set_split_settings(mode, train, val)

    def get_split_settings(self) -> Tuple[str, Union[float, List[str]], Union[float, List[str]]]:
        return self.split_manager.get_split_settings()

    @property
    def number_of_split_collections(self) -> int:
        return len(self.split_manager.get_split_collections().get("train", []))

    # Training-related methods
    def add_training_task(self, slug: str, task_id: int) -> None:
        self.training_manager.add_training_tasks(slug, [task_id])

    def add_training_tasks(self, slug: str, task_ids: List[int]) -> None:
        self.training_manager.add_training_tasks(slug, task_ids)

    def get_training_tasks(self) -> Dict[str, List[int]]:
        return self.training_manager.get_training_tasks()

    @property
    def training_tasks(self) -> Dict[str, List[int]]:
        return self.training_manager.get_training_tasks()

    # Background tasks-related methods
    def register_background_task(
        self, job_id: str, interval: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a background task for persistence

        Args:
            job_id: Unique task identifier
            interval: Interval in seconds
            metadata: Optional metadata for the task
        """
        self.background_tasks.add_task(job_id, interval, metadata)

    def update_background_task(self, job_id: str, **kwargs) -> bool:
        """
        Update an existing background task

        Args:
            job_id: Task identifier
            **kwargs: Properties to update (interval, enabled, metadata)

        Returns:
            bool: True if task was updated, False if not found
        """
        return self.background_tasks.update_task(job_id, **kwargs) is not None

    def record_task_execution(self, job_id: str) -> None:
        """
        Record that a task has been executed

        Args:
            job_id: Task identifier
        """
        self.background_tasks.record_task_run(job_id)

    def remove_background_task(self, job_id: str) -> bool:
        """
        Remove a background task

        Args:
            job_id: Task identifier

        Returns:
            bool: True if task was removed, False if not found
        """
        return self.background_tasks.remove_task(job_id)

    def enable_background_task(self, job_id: str, enabled: bool = True) -> bool:
        """
        Enable or disable a background task

        Args:
            job_id: Task identifier
            enabled: Whether task should be enabled

        Returns:
            bool: True if task was updated, False if not found
        """
        return self.background_tasks.enable_task(job_id, enabled)

    def get_background_task(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific background task by job_id

        Args:
            job_id: Task identifier

        Returns:
            Dict or None: Task data if found, None otherwise
        """
        task = self.background_tasks.get_task(job_id)
        return task.to_dict() if task else None

    def get_all_background_tasks(self) -> Dict[str, BackgroundTask]:
        """
        Get all background tasks

        Returns:
            Dict: Dictionary of task IDs to task data
        """
        return self.background_tasks.get_all_tasks()

    def get_all_background_tasks_json(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all background tasks

        Returns:
            Dict: Dictionary of task IDs to task data
        """
        return {job_id: task.to_dict() for job_id, task in self.get_all_background_tasks().items()}

    def get_enabled_background_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all enabled background tasks

        Returns:
            Dict: Dictionary of task IDs to task data for enabled tasks
        """
        return {
            job_id: task.to_dict()
            for job_id, task in self.background_tasks.get_enabled_tasks().items()
        }
