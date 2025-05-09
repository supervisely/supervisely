from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from supervisely.nn.active_learning.session import ActiveLearningSession

from supervisely.api.api import Api
from supervisely.nn.active_learning.scheduler.scheduler import SchedulerJobs
from supervisely.nn.active_learning.utils.constants import CLOUD_IMPORT_SLUG
from supervisely.sly_logger import logger


def import_from_cloud(al_session, path: str) -> dict:
    """
    Import data from cloud storage to input project

    Args:
        path: Path to data in cloud storage (e.g., 'provider://bucket-name/path/to/folder')

    Returns:
        dict: Information about the import task
    """
    al_session: ActiveLearningSession
    logger.info(f"Starting import from cloud storage: {path}")

    # Get the module ID for importing from cloud
    module_id = al_session.api.app.get_ecosystem_module_id(CLOUD_IMPORT_SLUG)
    module_info = al_session.api.app.get_ecosystem_module_info(module_id)

    # Prepare parameters for import
    params = module_info.get_arguments(images_project=al_session.state.project_id)
    params["slyFolder"] = path

    # Start import task
    session = al_session.api.app.start(
        agent_id=49,  # ! TODO: Make this configurable or automatically detect
        module_id=module_id,
        workspace_id=al_session.workspace_id,
        task_name="Import from Cloud Storage",
        params=params,
    )

    task_id = session.task_id
    logger.info(f"Cloud import started, task_id = {task_id}")

    # Add import task to state
    al_session.state.add_import_task(CLOUD_IMPORT_SLUG, task_id)

    return task_id


def schedule_cloud_import(al_session, path: str, interval: int = 60) -> int:
    """
    Schedule periodic import from cloud storage

    Args:
        path: Path to data in cloud storage
        interval: Interval between imports in seconds

    Returns:
        int: Scheduled job ID
    """
    al_session: ActiveLearningSession
    if interval <= 0:
        raise ValueError("Interval must be greater than 0 seconds")

    logger.info(f"Scheduling cloud import from {path} every {interval} seconds")

    # Schedule import job using the scheduler
    job_id = al_session.scheduler.add_job(
        job_id=SchedulerJobs.CLOUD_IMPORT,
        func=import_from_cloud,
        interval_sec=interval,
        args=(path,),
    )

    return job_id


def unschedule_cloud_import(al_session) -> bool:
    """
    Remove scheduled cloud import job

    Returns:
        bool: True if job was removed, False otherwise
    """
    al_session: ActiveLearningSession
    removed = al_session.scheduler.remove_job(SchedulerJobs.CLOUD_IMPORT)
    return removed
