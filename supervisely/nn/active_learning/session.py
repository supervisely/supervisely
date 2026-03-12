import random
from collections import defaultdict
from typing import Callable, Optional

from supervisely.api.api import Api
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.nn.active_learning.importing.importer import ActiveLearningImporter
from supervisely.nn.active_learning.labeling.labeling_service import LabelingService
from supervisely.nn.active_learning.sampling.sampler import ActiveLearningSampler
from supervisely.nn.active_learning.scheduler.scheduler import (
    PersistentTasksScheduler,
    SchedulerJobs,
)
from supervisely.nn.active_learning.state.state import ALState
from supervisely.nn.active_learning.utils.constants import EMBEDDINGS_GENERATOR_SLUG
from supervisely.sly_logger import logger


class ActiveLearningSession:
    INPUT_PROJECT_NAME = "Input Project"
    LABELING_PROJECT_NAME = "Labeling Project"
    TRAINING_PROJECT_NAME = "Training Project"
    TRAIN_COLLECTION_PREFIX = "train_"
    VAL_COLLECTION_PREFIX = "val_"
    """
    Class to manage Active Learning session in Supervisely Solution.
    This class handles all operations related to the Solution's active learning process:
        - Managing state (collect and update projects IDs, import tasks, labeling tasks, etc.)
        - Sampling tasks (comparing projects, sampling images using different methods and copying them to labeling project)
        - Managing import tasks (importing images to the input project)
        - Labeling tasks (creating labeling queue, managing labeling process)
        - Managing train/val split settings
        - Managing training tasks (creating training iterations, managing training process and results)
    """

    def __init__(self, api: Api, project_id=None, workspace_id=None, team_id=None):
        if project_id is None and workspace_id is None and team_id is None:
            raise ValueError("Either project_id or workspace_id or team_id must be provided.")
        self.api = api
        if project_id is not None:
            self.state = ALState(api, project_id)
        elif workspace_id is not None:
            self.state = ALState.create_from_workspace(api, workspace_id, self.INPUT_PROJECT_NAME)
        else:
            self.state = ALState.create_from_team(api, team_id, self.INPUT_PROJECT_NAME)
        self.project_id = self.state.project_id
        self.workspace_id = self.state.project.workspace_id
        self.team_id = self.state.project.team_id

        self.scheduler = PersistentTasksScheduler(self.state)
        # * to be implemented
        # # Map job IDs to functions
        # job_functions = {
        #     SchedulerJobs.LABELING_QUEUE_STATS: check_labeling_info,
        #     SchedulerJobs.CLOUD_IMPORT: import_from_cloud,
        # }

        # # Restore previously scheduled jobs (after app restart)
        # restored_jobs = scheduler.restore_jobs(job_functions)
        # print(f"Restored jobs: {restored_jobs}")

        self.team_id = self.state.project_manager.get_project().team_id
        self.workspace_id = self.state.project_manager.get_project().workspace_id

        self.importer = ActiveLearningImporter(self)
        self.sampler = ActiveLearningSampler(self)
        self.labeling_service = LabelingService(self)

    def send_project_to_embedding_generator(self):
        module_info = self.api.app.get_ecosystem_module_info(slug=EMBEDDINGS_GENERATOR_SLUG)
        sessions = self.api.app.get_sessions(
            self.team_id, module_info.id, statuses=[self.api.task.Status.STARTED]
        )
        if len(sessions) == 0:
            logger.error("No active sessions found for embeddings generator.")
            return
        session = sessions[0]
        # g.api.app.wait(session.task_id, target_status=g.api.task.Status.STARTED)
        logger.info(f"Session is ready for API calls: {session.task_id} (embeddings generator)")
        return self.api.app.send_request(
            session.task_id, "embeddings", data={"project_id": self.project_id}
        )

    def is_refresh_project_info_scheduled(self) -> bool:
        """
        Check if the project information refresh task is scheduled.
        """
        return self.scheduler.is_job_scheduled(SchedulerJobs.REFRESH_PROJECT_INFO)

    def schedule_refresh_project_info(self, func, interval: int = 20) -> None:
        """
        Schedule a job to refresh project information at a specified interval.
        """
        if interval <= 0:
            raise ValueError("Interval must be greater than 0 seconds")
        self.scheduler.add_job(SchedulerJobs.REFRESH_PROJECT_INFO, func, interval)

    def restore_scheduled_refresh_project_info(self, func: Callable) -> None:
        """
        Restore scheduled job if exists.
        """
        self.scheduler.restore_jobs({SchedulerJobs.REFRESH_PROJECT_INFO: func})
        logger.info("Project info refresh job restored")

    def import_from_cloud(self, path: str) -> None:
        """
        Import images from cloud storage to the input project.
        """
        return self.importer.import_from_cloud(path)

    def wait_import_completion(self, task_id: int) -> bool:
        """
        Wait for import task to complete and return status.
        """
        return self.importer.wait_import_completion(task_id)

    def sample(self, sampling_settings: dict) -> None:
        """
        Sample images from the input project and copy them to the labeling project.
        """
        return self.sampler.sample(settings=sampling_settings)

    def preview_sampled_images(self, sample_settings: dict, limit: int = 6) -> list:
        """
        Preview sampled images.
        """
        return self.sampler.preview(settings=sample_settings, limit=limit)

    def get_labeling_stats(self) -> None:
        """
        Get labeling stats.
        """
        return self.labeling_service.get_labeling_stats()

    def move_labeled_images_to_training_project(self, min_batch: Optional[int] = None) -> None:
        """
        Move labeled images to the training project.
        """
        return self.labeling_service.move_to_training_project(min_batch=min_batch)

    def schedule_refresh_labeling_info(self, func, interval: int = 20) -> None:
        """
        Schedule a job to refresh labeling information at a specified interval.
        """
        self.labeling_service.schedule_refresh(func=func, interval=interval)

    def is_labeling_info_scheduled(self) -> bool:
        """
        Check if the labeling information refresh task is scheduled.
        """
        return self.labeling_service.is_refresh_scheduled()

    def restore_scheduled_refresh_labeling_info(self, func: Callable) -> None:
        """
        Restore scheduled job if exists.
        """
        self.labeling_service.restore_scheduled_refresh(func=func)

    def schedule_move_to_training_project(
        self, func, interval: int = 20, min_batch: Optional[int] = None
    ) -> None:
        """
        Schedule a job to move labeled images to the training project at a specified interval.
        """
        self.labeling_service.schedule_move_to_training_project(
            func=func, interval=interval, min_batch=min_batch
        )

    def unschedule_move_to_training_project(self) -> None:
        """
        Unschedule the job to move labeled images to the training project.
        """
        self.labeling_service.unschedule_move_to_training_project()

    def restore_scheduled_move_to_training_project(self, func: Callable) -> None:
        """
        Restore scheduled job if exists.
        """
        self.labeling_service.restore_scheduled_move_to_training_project(func=func)

    def is_move_to_training_project_scheduled(self) -> bool:
        """
        Check if the move to training project task is scheduled.
        """
        return self.labeling_service.is_move_to_training_project_scheduled()

    def add_annotators_to_labeling_queue(self, annotators: list) -> None:
        """
        Add annotators to the labeling queue.
        """
        self.labeling_service.add_annotators(annotators)

    def add_reviewers_to_labeling_queue(self, reviewers: list) -> None:
        """
        Add reviewers to the labeling queue.
        """
        self.labeling_service.add_reviewers(reviewers)
