from typing import Callable, List, Optional, Union

from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.project import read_any_single_project
from supervisely.project.pointcloud_episode_project import (
    upload_pointcloud_episode_project,
)
from supervisely.project.pointcloud_project import upload_pointcloud_project
from supervisely.project.project import upload_project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import upload_video_project
from supervisely.project.volume_project import upload_volume_project


def upload(
    src_dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: Optional[bool] = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    """
    Uploads project of any type from the local directory. See methods `sly.upload_project`,
    `sly.upload_video_project`, `sly.upload_volume_project`, `sly.upload_pointcloud_project`,
    `sly.upload_pointcloud_episode_project` to examine full list of possible arguments.

    :param src_dir: Source path to local directory.
    :type src_dir: str
    :param api: Supervisely API address and token.
    :type api: Api
    :param workspace_id: Destination workspace ID.
    :type workspace_id: int
    :param project_name: Custom project name. By default, it's a directory name.
    :type project_name: str, optional
    :param log_progress: Show uploading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking upload progress.
    :type progress_cb: tqdm or callable, optional

    :return: None.
    :rtype: NoneType
    :Usage example:

    .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        src_dir = 'your/local/source/dir'

        # Upload image project
        project_fs = sly.read_any_single_project(src_dir)
        num_images = project_fs.total_items

        pbar = tqdm(desc="Uploading image project", total=num_images)
        sly.upload(src_dir, api, workspace_id, project_name, progress_cb=pbar)

        # Upload video project
        sly.upload(
            src_dir,
            api,
            workspace_id,
            project_name="Some Video Project",
            log_progress=True,
            include_custom_data=True
        )

        # Upload volume project
        sly.upload(src_dir, api, workspace_id, project_name="Some Volume Project", log_progress=True)

        # Upload pointcloud project
        project_fs = read_any_single_project(directory)
        num_ptcl = project_fs.items_count

        pbar = tqdm(desc="Uploading pointcloud project", total=num_ptcl)
        sly.upload(
            src_dir,
            api,
            workspace_id,
            project_name="Some Pointcloud Project",
            progress_cb=pbar,
        )

        # Upload pointcloud episodes project
        project_fs = read_any_single_project(src_dir)
        num_ptclep = project_fs.items_count

        with tqdm(desc="Upload pointcloud episodes project", total=num_ptclep) as pbar:
            sly.upload(
                src_dir,
                api,
                workspace_id,
                project_name="Some Pointcloud Episodes Project",
                progress_cb=pbar,
            )
    """

    project_fs = read_any_single_project(src_dir)

    if progress_cb:
        log_progress = False

    if project_fs.meta.project_type == ProjectType.IMAGES.value:
        upload_project(
            dir=src_dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif project_fs.meta.project_type == ProjectType.VIDEOS.value:
        if progress_cb:
            log_progress = True
        upload_video_project(
            dir=src_dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            **kwargs,
        )
    elif project_fs.meta.project_type == ProjectType.VOLUMES.value:
        if progress_cb:
            log_progress = True
        upload_volume_project(
            dir=src_dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
        )

    elif project_fs.meta.project_type == ProjectType.POINT_CLOUDS.value:
        upload_pointcloud_project(
            directory=src_dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    elif project_fs.meta.project_type == ProjectType.POINT_CLOUD_EPISODES.value:
        upload_pointcloud_episode_project(
            directory=src_dir,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    else:
        raise ValueError(f"Unknown type of project ({project_fs.meta.project_type})")
