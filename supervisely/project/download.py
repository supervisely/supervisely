from typing import List, Optional, Callable, Union
from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.project.project import download_project
from supervisely.project.pointcloud_episode_project import download_pointcloud_episode_project
from supervisely.project.pointcloud_project import download_pointcloud_project
from supervisely.project.video_project import download_video_project
from supervisely.project.volume_project import download_volume_project
from supervisely.project.project_type import ProjectType


def download(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[Callable, tqdm]] = None,
) -> None:
    # TODO configure arguments for nested functions

    project_info = api.project.get_info_by_id(project_id)

    if project_info.type == ProjectType.IMAGES.value:
        download_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif project_info.type == ProjectType.VIDEOS.value:
        download_video_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif project_info.type == ProjectType.VOLUMES.value:
        download_volume_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
        )  # TODO

    elif project_info.type == ProjectType.POINT_CLOUDS.value:
        download_pointcloud_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif project_info.type == ProjectType.POINT_CLOUD_EPISODES.value:
        download_pointcloud_episode_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    else:
        raise ValueError(f"Unknown type of project ({project_info.type})")
