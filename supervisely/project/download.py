import os
from typing import Callable, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely import get_project_class
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.io.fs import copy_dir_recursively, dir_exists

CACHE_DIR = "/apps_cache"


def download(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    """
    Downloads project of any type to the local directory. See methods `sly.download_project`,
    `sly.download_video_project`, `sly.download_volume_project`, `sly.download_pointcloud_project`,
    `sly.download_pointcloud_episode_project` to examine full list of possible arguments.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking download progress.
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

        dest_dir = 'your/local/dest/dir'

        # Download image project
        project_id_image = 17732
        project_info = api.project.get_info_by_id(project_id_image)
        num_images = project_info.items_count

        p = tqdm(desc="Downloading image project", total=num_images)
        sly.download(
            api,
            project_id_image,
            dest_dir,
            progress_cb=p,
            save_image_info=True,
            save_images=True,
        )

        # Download video project
        project_id_video = 60498
        project_info = api.project.get_info_by_id(project_id_video)
        num_videos = project_info.items_count

        p = tqdm(desc="Downloading video project", total=num_videos)
        sly.download(
            api,
            project_id_video,
            dest_dir,
            progress_cb=p,
            save_video_info=True,
        )

        # Download volume project
        project_id_volume = 18594
        project_info = api.project.get_info_by_id(project_id_volume)
        num_volumes = project_info.items_count

        p = tqdm(desc="Downloading volume project",total=num_volumes)
        sly.download(
            api,
            project_id_volume,
            dest_dir,
            progress_cb=p,
            download_volumes=True,
        )

        # Download pointcloud project
        project_id_ptcl = 18592
        project_info = api.project.get_info_by_id(project_id_ptcl)
        num_ptcl = project_info.items_count

        p = tqdm(desc="Downloading pointcloud project", total=num_ptcl)
        sly.download(
            api,
            project_id_ptcl,
            dest_dir,
            progress_cb=p,
            download_pointclouds_info=True,
        )

        # Download some datasets from pointcloud episodes project
        project_id_ptcl_ep = 18593
        dataset_ids = [43546, 45765, 45656]

        p = tqdm(
            desc="Download some datasets from pointcloud episodes project",
            total=len(dataset_ids),
        )
        sly.download(
            api,
            project_id_ptcl_ep,
            dest_dir,
            dataset_ids,
            progress_cb=p,
            download_pcd=True,
            download_related_images=True,
            download_annotations=True,
            download_pointclouds_info=True,
        )
    """

    project_info = api.project.get_info_by_id(project_id)

    if progress_cb is not None:
        log_progress = False

    project_class = get_project_class(project_info.type)

    project_class.download(
        api=api,
        project_id=project_id,
        dest_dir=dest_dir,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        **kwargs,
    )


def _get_cache_dir(project_id: int, dataset_name: str = None) -> str:
    p = os.path.join(CACHE_DIR, str(project_id))
    if dataset_name is not None:
        os.path.join(p, dataset_name)
    return p


def is_cached(project_id, dataset_name: str = None) -> bool:
    return dir_exists(_get_cache_dir(project_id, dataset_name))


def _split_by_cache(project_id: int, dataset_names: List[str]) -> Tuple[List, List]:
    if not is_cached(project_id):
        return dataset_names, []
    to_download = [ds_name for ds_name in dataset_names if not is_cached(project_id, ds_name)]
    cached = [ds_name for ds_name in dataset_names if is_cached(project_id, ds_name)]
    return to_download, cached


def download_to_cache(
    api: Api,
    project_id: int,
    dataset_infos: List[DatasetInfo] = None,
    dataset_ids: List[int] = None,
    log_progress: bool = True,
    progress_cb=None,
    **kwargs,
) -> Tuple[List, List]:
    """
    Download datasets to cache.
    If dataset_infos is not None, dataset_ids must be None and vice versa.
    If both dataset_infos and dataset_ids are None, all datasets of the project will be downloaded.
    :param api: supervisely.api.api.Api
    :param project_id: int
    :param dataset_infos: List[supervisely.api.dataset_api.DatasetInfo]
    :param dataset_ids: List[int]
    :param log_progress: bool
    :param progress_cb: callable which will be called with number of items downloaded
    :return: Tuple[List, List] where the first list contains names of downloaded datasets and the second list contains
    names of cached datasets
    """
    cache_project_dir = _get_cache_dir(project_id)
    if dataset_infos is not None and dataset_ids is not None:
        raise ValueError("dataset_infos and dataset_ids cannot be specified at the same time")
    if dataset_infos is None:
        if dataset_ids is None:
            dataset_infos = api.dataset.get_list(project_id)
        else:
            dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
    name_to_info = {info.name: info for info in dataset_infos}
    to_download, cached = _split_by_cache(project_id, [info.name for info in dataset_infos])
    if progress_cb is not None:
        cached_items_n = sum(name_to_info[ds_name].items_count for ds_name in cached)
        progress_cb(cached_items_n)
    dataset_ids = [name_to_info[name].id for name in to_download]
    download(
        api=api,
        project_id=project_id,
        dest_dir=cache_project_dir,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        **kwargs,
    )
    return to_download, cached


def copy_from_cache(
    project_id: int, dst_dir: str, dataset_name: str = None, progress_cb: Callable = None
):
    """
    Copy project or dataset from cache to the specified directory.
    If dataset_name is None, the whole project will be copied.
    :param project_id: int
    :param dst_dir: str
    :param dataset_name: str
    :param progress_cb: callable
    :return: None
    """
    if not is_cached(project_id, dataset_name):
        msg = f"Project {project_id} is not cached"
        if dataset_name is not None:
            msg = f"Dataset {dataset_name} of project {project_id} is not cached"
        raise RuntimeError(msg)
    cache_dir = _get_cache_dir(project_id, dataset_name)
    copy_dir_recursively(cache_dir, dst_dir, progress_cb)


def download_using_cache(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: Optional[bool] = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    """ """
    download_to_cache(
        api,
        project_id,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        **kwargs,
    )
    copy_from_cache(project_id, dest_dir, progress_cb=progress_cb)
