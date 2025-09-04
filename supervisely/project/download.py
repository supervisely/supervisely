import asyncio
import os
from typing import Callable, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely import get_project_class
from supervisely._utils import run_coroutine
from supervisely.annotation.annotation import Annotation, ProjectMeta
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.io.env import apps_cache_dir
from supervisely.io.fs import (
    copy_dir_recursively,
    copy_file,
    dir_exists,
    get_directory_size,
    remove_dir,
)
from supervisely.io.json import load_json_file
from supervisely.project import Project
from supervisely.project.project import Dataset, OpenMode, ProjectType
from supervisely.sly_logger import logger


def download(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
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
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool
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
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

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

        # Download video project with automatic logging...
        sly.download(
            api,
            project_id_video,
            dest_dir,
            save_video_info=True,
        )
        # ...or disable logging at all
        sly.download(
            api,
            project_id_video,
            dest_dir,
            log_progress=False,
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


def download_async(
    api: Api,
    project_id: int,
    dest_dir: str,
    semaphore: Optional[asyncio.Semaphore] = None,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    **kwargs,
) -> None:
    project_info = api.project.get_info_by_id(project_id)

    if progress_cb is not None:
        log_progress = False

    project_class = get_project_class(project_info.type)
    if hasattr(project_class, "download_async"):
        download_coro = project_class.download_async(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            semaphore=semaphore,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
            **kwargs,
        )
        run_coroutine(download_coro)
    else:
        raise NotImplementedError(f"Method download_async is not implemented for {project_class}")


def download_async_or_sync(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
):
    """
    Download project asynchronously if possible, otherwise download synchronously.
    Automatically detects project type.
    You can pass :class:`ProjectInfo` as `project_info` kwarg to avoid additional API requests.

    In case of error during asynchronous download, the function will switch to synchronous download.
    """
    project_info = kwargs.pop("project_info", None)
    if not isinstance(project_info, ProjectInfo) or project_info.id != project_id:
        project_info = api.project.get_info_by_id(project_id)

    if progress_cb is not None:
        log_progress = False

    project_class = get_project_class(project_info.type)

    switch_to_sync = False
    if hasattr(project_class, "download_async"):
        try:
            download_coro = project_class.download_async(
                api=api,
                project_id=project_id,
                dest_dir=dest_dir,
                semaphore=semaphore,
                dataset_ids=dataset_ids,
                log_progress=log_progress,
                progress_cb=progress_cb,
                **kwargs,
            )
            run_coroutine(download_coro)
        except Exception as e:
            if kwargs.get("resume_download", False) is False:
                remove_dir(dest_dir)
            logger.error(f"Failed to download project {project_id} asynchronously: {e}")
            logger.warning("Switching to synchronous download")
            switch_to_sync = True
    else:
        switch_to_sync = True

    if switch_to_sync:
        project_class.download(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
            **kwargs,
        )


def download_fast(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
) -> None:
    """
    Download project in a fast mode.
    Items are downloaded asynchronously. If an error occurs, the method will fallback to synchronous download.
    Automatically detects project type.

    :param api: Supervisely API address and token.
    :type api: :class:`Api<supervisely.api.api.Api>`
    :param project_id: Supervisely downloadable project ID.
    :type project_id: :class:`int`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`
    :param dataset_ids: Filter datasets by IDs.
    :type dataset_ids: :class:`list` [ :class:`int` ], optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking download progress.
    :type progress_cb: tqdm or callable, optional
    :param semaphore: Semaphore to limit the number of concurrent downloads of items.
    :type semaphore: :class:`asyncio.Semaphore`, optional
    :param only_image_tags: Download project with only images tags (without objects tags).
    :type only_image_tags: :class:`bool`, optional
    :param save_image_info: Download images infos or not.
    :type save_image_info: :class:`bool`, optional
    :param save_images: Download images or not.
    :type save_images: :class:`bool`, optional
    :param save_image_meta: Download images metadata in JSON format or not.
    :type save_image_meta: :class:`bool`, optional
    :param images_ids: Filter images by IDs.
    :type images_ids: :class:`list` [ :class:`int` ], optional
    :param resume_download: Resume download enables to download only missing files avoiding erase of existing files.
    :type resume_download: :class:`bool`, optional
    :param switch_size: Size threshold that determines how an item will be downloaded.
                        Items larger than this size will be downloaded as single files, while smaller items will be downloaded as a batch.
                        Useful for projects with different item sizes and when you exactly know which size will perform better with batch download.
    :type switch_size: :class:`int`, optional
    :param batch_size: Number of items to download in a single batch.
    :type batch_size: :class:`int`, optional
    :param download_blob_files: Download project with Blob files in native format.
                                If False - download project like a regular project in classic Supervisely format.
    :type download_blob_files: :class:`bool`, optional
    :param project_info: Project info object. To avoid additional API requests.
    :type project_info: :class:`ProjectInfo`, optional
    :param skip_create_readme: Skip creating README.md file. Default is False.
    :type skip_create_readme: bool, optional
    :return: None
    :rtype: NoneType

    :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 8888
            save_directory = "/path/to/save/projects"

            sly.download_fast(api, project_id, save_directory)

    """

    download_async_or_sync(
        api=api,
        project_id=project_id,
        dest_dir=dest_dir,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        semaphore=semaphore,
        **kwargs,
    )


def _get_cache_dir(project_id: int, dataset_path: str = None) -> str:
    p = os.path.join(apps_cache_dir(), str(project_id))
    if dataset_path is not None:
        p = os.path.join(p, dataset_path)
    return p


def is_cached(project_id, dataset_path: str = None) -> bool:
    return dir_exists(_get_cache_dir(project_id, dataset_path))


def _split_by_cache(project_id: int, dataset_paths: List[str]) -> Tuple[List, List]:
    if not is_cached(project_id):
        return dataset_paths, []
    to_download = [ds_path for ds_path in dataset_paths if not is_cached(project_id, ds_path)]
    cached = [ds_path for ds_path in dataset_paths if is_cached(project_id, ds_path)]
    return to_download, cached


def get_cache_size(project_id: int, dataset_path: str = None) -> int:
    if not is_cached(project_id, dataset_path):
        return 0
    cache_dir = _get_cache_dir(project_id, dataset_path)
    return get_directory_size(cache_dir)


def _get_items_infos(api: Api, project_type: str, dataset_id: int) -> List[ImageInfo]:
    funcs = {
        str(ProjectType.IMAGES): api.image.get_list,
        str(ProjectType.VIDEOS): api.video.get_list,
        str(ProjectType.POINT_CLOUDS): api.pointcloud.get_list,
        str(ProjectType.POINT_CLOUD_EPISODES): api.pointcloud.get_list,
        str(ProjectType.VOLUMES): api.volume.get_list,
    }
    return funcs[project_type](dataset_id)


def _project_meta_changed(meta1: ProjectMeta, meta2: ProjectMeta) -> bool:
    if len(meta1.obj_classes) != len(meta2.obj_classes) or len(meta1.tag_metas) != len(
        meta2.tag_metas
    ):
        return True
    for obj_class1 in meta1.obj_classes:
        obj_class2 = meta2.get_obj_class(obj_class1.name)
        if obj_class2 is None or obj_class1 != obj_class2 or obj_class1.sly_id != obj_class2.sly_id:
            return True
    for tag_meta1 in meta1.tag_metas:
        tag_meta2 = meta2.get_tag_meta(tag_meta1.name)
        if tag_meta2 is None or tag_meta1 != tag_meta2 or tag_meta1.sly_id != tag_meta2.sly_id:
            return True
    return False


def _get_ds_full_name(
    dataset_info: DatasetInfo, all_ds_infos: List[DatasetInfo], suffix: str = ""
) -> str:
    if dataset_info.parent_id is None:
        return dataset_info.name + suffix
    parent = next((ds_info for ds_info in all_ds_infos if ds_info.id == dataset_info.parent_id))
    return _get_ds_full_name(parent, all_ds_infos, "/" + dataset_info.name + suffix)


def _validate_dataset(
    api: Api,
    project_id: int,
    project_type: str,
    project_meta: ProjectMeta,
    dataset_info: DatasetInfo,
    all_ds_infos: List[DatasetInfo] = None,
):
    try:
        project_class = get_project_class(project_type)
        project: Project = project_class(_get_cache_dir(project_id), OpenMode.READ)
    except Exception:
        logger.debug("Validating dataset failed. Error reading project.", exc_info=True)
        return False
    try:
        items_infos_dict = {
            item_info.name: item_info
            for item_info in _get_items_infos(api, project_type, dataset_info.id)
        }
    except:
        logger.debug("Validating dataset failed. Unable to download items infos.", exc_info=True)
        return False
    if all_ds_infos is None:
        all_ds_infos = api.dataset.get_list(project_id, recursive=True)
    project_meta_changed = _project_meta_changed(project_meta, project.meta)
    for dataset in project.datasets:
        dataset: Dataset
        if dataset.name == _get_ds_full_name(dataset_info, all_ds_infos):
            diff = set(items_infos_dict.keys()).difference(set(dataset.get_items_names()))
            if diff:
                logger.debug(
                    "Validating dataset failed. Items are missing.",
                    extra={"missing_items": diff},
                )
                return False
            for item_name, _, ann_path in dataset.items():
                try:
                    item_info = dataset.get_item_info(item_name)
                except Exception:
                    logger.debug(
                        "Validating dataset failed. Error reading item info.",
                        extra={"item_name": item_name},
                        exc_info=True,
                    )
                    return False
                if item_info.name not in items_infos_dict:
                    logger.debug(
                        "Validating dataset failed. Item info is redundant.",
                        extra={"item_name": item_name},
                    )
                    return False
                if item_info != items_infos_dict[item_info.name]:
                    logger.debug(
                        "Validating dataset failed. Item info is different.",
                        extra={"item_name": item_name},
                    )
                    return False
                if project_meta_changed:
                    try:
                        Annotation.from_json(load_json_file(ann_path), project_meta)
                    except Exception:
                        logger.debug(
                            "Validating dataset failed. Error reading annotation.",
                            extra={"item_name": item_name},
                            exc_info=True,
                        )
                        return False
            return True
    logger.debug(
        "Validating dataset failed. Dataset is missing.", extra={"dataset_name": dataset_info.name}
    )
    return False


def _validate(
    api: Api,
    project_info: ProjectInfo,
    project_meta: ProjectMeta,
    dataset_infos: List[DatasetInfo],
    all_ds_infos: List[DatasetInfo] = None,
):
    project_id = project_info.id
    to_download, cached = _split_by_cache(
        project_id, [get_dataset_path(api, dataset_infos, info.id) for info in dataset_infos]
    )
    to_download, cached = set(to_download), set(cached)
    for dataset_info in dataset_infos:
        ds_path = get_dataset_path(api, dataset_infos, dataset_info.id)
        if ds_path in to_download:
            continue
        if not _validate_dataset(
            api,
            project_id,
            project_info.type,
            project_meta,
            dataset_info,
            all_ds_infos,
        ):
            to_download.add(ds_path)
            cached.remove(ds_path)
            logger.info(
                f"Dataset {ds_path} of project {project_id} is not up to date and will be re-downloaded."
            )
    return list(to_download), list(cached)


def _add_save_items_infos_to_kwargs(kwargs: dict, project_type: str):
    arg_name = {
        str(ProjectType.IMAGES): "save_image_info",
        str(ProjectType.VIDEOS): "save_video_info",
        str(ProjectType.POINT_CLOUDS): "download_pointclouds_info",
        str(ProjectType.POINT_CLOUD_EPISODES): "download_pointclouds_info",
        str(ProjectType.VOLUMES): "save_volumes_info",
    }
    kwargs[arg_name[project_type]] = True
    return kwargs


def _add_resume_download_to_kwargs(kwargs: dict, project_type: str):
    supported_force_projects = (str(ProjectType.IMAGES), (str(ProjectType.VIDEOS)))
    if project_type in supported_force_projects:
        kwargs["resume_download"] = True
    return kwargs


def _download_project_to_cache(
    api: Api,
    project_info: ProjectInfo,
    dataset_infos: List[DatasetInfo],
    log_progress: bool = True,
    progress_cb: Callable = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
):
    project_id = project_info.id
    project_type = project_info.type
    kwargs = _add_save_items_infos_to_kwargs(kwargs, project_type)
    kwargs = _add_resume_download_to_kwargs(kwargs, project_type)
    cached_project_dir = _get_cache_dir(project_id)
    if len(dataset_infos) == 0:
        logger.debug("No datasets to download")
        return
    download_fast(
        api=api,
        project_id=project_id,
        dest_dir=cached_project_dir,
        dataset_ids=[info.id for info in dataset_infos],
        log_progress=log_progress,
        progress_cb=progress_cb,
        semaphore=semaphore,
        **kwargs,
    )


def download_to_cache(
    api: Api,
    project_id: int,
    dataset_infos: List[DatasetInfo] = None,
    dataset_ids: List[int] = None,
    log_progress: bool = True,
    progress_cb=None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
) -> Tuple[List, List]:
    """
    Download datasets to cache.
    If dataset_infos is not None, dataset_ids must be None and vice versa.
    If both dataset_infos and dataset_ids are None, all datasets of the project will be downloaded.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dataset_infos: Specified list of Dataset Infos which will be downloaded.
    :type dataset_infos: list(DatasetInfo), optional
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool, optional
    :param progress_cb: Function for tracking download progress. Will be called with number of items downloaded.
    :type progress_cb: tqdm or callable, optional
    :param semaphore: Semaphore for limiting the number of concurrent downloads if using async download.

    :return: Tuple where the first list contains names of downloaded datasets and the second list contains
    names of cached datasets
    :rtype: Tuple[List, List]
    """
    project_info = api.project.get_info_by_id(project_id)
    project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    if dataset_infos is not None and dataset_ids is not None:
        raise ValueError("dataset_infos and dataset_ids cannot be specified at the same time")
    all_ds_infos = api.dataset.get_list(project_id, recursive=True)
    if dataset_infos is None:
        if dataset_ids is None:
            dataset_infos = all_ds_infos
        else:
            dataset_infos = [ds_info for ds_info in all_ds_infos if ds_info.id in dataset_ids]
    path_to_info = {get_dataset_path(api, dataset_infos, info.id): info for info in dataset_infos}
    to_download, cached = _validate(api, project_info, project_meta, dataset_infos, all_ds_infos)
    if progress_cb is not None:
        cached_items_n = sum(path_to_info[ds_path].items_count for ds_path in cached)
        progress_cb(cached_items_n)
    _download_project_to_cache(
        api=api,
        project_info=project_info,
        dataset_infos=[path_to_info[ds_path] for ds_path in to_download],
        log_progress=log_progress,
        progress_cb=progress_cb,
        semaphore=semaphore,
        **kwargs,
    )
    return to_download, cached


def _get_dataset_parents(api: Api, dataset_infos: List[DatasetInfo], dataset_id):
    dataset_infos_dict = {info.id: info for info in dataset_infos}
    this_dataset_info = dataset_infos_dict.get(dataset_id, None)
    if this_dataset_info is None:
        this_dataset_info = api.dataset.get_info_by_id(dataset_id)
    if this_dataset_info.parent_id is None:
        return []
    parent = _get_dataset_parents(
        api, list(dataset_infos_dict.values()), this_dataset_info.parent_id
    )
    this_parent = dataset_infos_dict.get(this_dataset_info.parent_id, None)
    if this_parent is None:
        this_parent = api.dataset.get_info_by_id(this_dataset_info.parent_id)
    return [*parent, this_parent.name]


def get_dataset_path(api: Api, dataset_infos: List[DatasetInfo], dataset_id: int) -> str:
    parents = _get_dataset_parents(api, dataset_infos, dataset_id)
    dataset_infos_dict = {info.id: info for info in dataset_infos}
    this_dataset_info = dataset_infos_dict.get(dataset_id, None)
    if this_dataset_info is None:
        this_dataset_info = api.dataset.get_info_by_id(dataset_id)
    return Dataset._get_dataset_path(this_dataset_info.name, parents)


def copy_from_cache(
    project_id: int,
    dest_dir: str,
    dataset_names: List[str] = None,
    progress_cb: Callable = None,
    dataset_paths: List[str] = None,
):
    """
    Copy project or dataset from cache to the specified directory.
    If dataset_name is None, the whole project will be copied.

    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_name: List of dataset paths to copy. If not specified, the whole project will be copied.
    :type dataset_name: str, optional
    :param progress_cb: Function for tracking copying progress. Will be called with number of bytes copied.
    :type progress_cb: tqdm or callable, optional
    :param dataset_paths: List of dataset paths to copy. If not specified, all datasets will be copied.
    :type dataset_paths: list(str), optional

    :return: None.
    :rtype: NoneType
    """
    if not is_cached(project_id):
        raise RuntimeError(f"Project {project_id} is not cached")
    if dataset_names is not None or dataset_paths is not None:
        if dataset_names is not None:
            dataset_paths = dataset_names
        for dataset_path in dataset_paths:
            if not is_cached(project_id, dataset_path):
                raise RuntimeError(f"Dataset {dataset_path} of project {project_id} is not cached")
    cache_dir = _get_cache_dir(project_id)
    if dataset_paths is None:
        copy_dir_recursively(cache_dir, dest_dir, progress_cb)
    else:
        # copy meta
        copy_file(os.path.join(cache_dir, "meta.json"), os.path.join(dest_dir, "meta.json"))
        # copy datasets
        for dataset_path in dataset_paths:
            copy_dir_recursively(
                os.path.join(cache_dir, dataset_path),
                os.path.join(dest_dir, dataset_path),
                progress_cb,
            )


def download_using_cache(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
) -> None:
    """
    Download project to the specified directory using cache.
    If dataset_ids is None, all datasets of the project will be downloaded.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded.
    :type dataset_ids: list(int), optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool
    :param progress_cb: Function for tracking download progress. Will be called with number of items downloaded.
    :type progress_cb: tqdm or callable, optional
    :param semaphore: Semaphore for limiting the number of concurrent downloads if using async download.
    :type semaphore: asyncio.Semaphore, optional

    :return: None.
    :rtype: NoneType
    """
    downloaded, cached = download_to_cache(
        api,
        project_id,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
        semaphore=semaphore,
        **kwargs,
    )
    copy_from_cache(project_id, dest_dir, [*downloaded, *cached])


def read_from_cached_project(
    project_id: int, dataset_name: str, image_names: List[int]
) -> List[Tuple[str, str]]:
    """
    Read images from cached project.

    :param project_id: Project ID.
    :type project_id: int
    :param dataset_name: Name of the dataset.
    :type dataset_name: str
    :param image_ids: List of image IDs.
    :type image_ids: list(int)

    :return: List of tuples of image path and annotation path.
    :rtype: list(str)
    """
    if not is_cached(project_id, dataset_name):
        raise RuntimeError(f"Dataset {dataset_name} of project {project_id} is not cached")

    dataset = Dataset(_get_cache_dir(project_id, dataset_name), OpenMode.READ)
    paths = []
    for image_name in image_names:
        image_path, ann_path = dataset.get_item_paths(image_name)
        paths.append((image_path, ann_path))
    return paths
