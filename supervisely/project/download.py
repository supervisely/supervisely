import os
import shutil
from typing import Callable, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely import get_project_class
from supervisely._utils import rand_str
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
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
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


def _get_cache_dir(project_id: int, dataset_name: str = None) -> str:
    p = os.path.join(apps_cache_dir(), str(project_id))
    if dataset_name is not None:
        p = os.path.join(p, dataset_name)
    return p


def is_cached(project_id, dataset_name: str = None) -> bool:
    return dir_exists(_get_cache_dir(project_id, dataset_name))


def _split_by_cache(project_id: int, dataset_names: List[str]) -> Tuple[List, List]:
    if not is_cached(project_id):
        return dataset_names, []
    to_download = [ds_name for ds_name in dataset_names if not is_cached(project_id, ds_name)]
    cached = [ds_name for ds_name in dataset_names if is_cached(project_id, ds_name)]
    return to_download, cached


def get_cache_size(project_id: int, dataset_name: str = None) -> int:
    if not is_cached(project_id, dataset_name):
        return 0
    cache_dir = _get_cache_dir(project_id, dataset_name)
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


def _validate_dataset(
    api: Api,
    project_id: int,
    project_type: str,
    project_meta: ProjectMeta,
    dataset_info: DatasetInfo,
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
    project_meta_changed = _project_meta_changed(project_meta, project.meta)
    for dataset in project.datasets:
        dataset: Dataset
        if dataset.name == dataset_info.name:
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
    api: Api, project_info: ProjectInfo, project_meta: ProjectMeta, dataset_infos: List[DatasetInfo]
):
    project_id = project_info.id
    to_download, cached = _split_by_cache(project_id, [info.name for info in dataset_infos])
    to_download, cached = set(to_download), set(cached)
    for dataset_info in dataset_infos:
        if dataset_info.name in to_download:
            continue
        if not _validate_dataset(
            api,
            project_id,
            project_info.type,
            project_meta,
            dataset_info,
        ):
            to_download.add(dataset_info.name)
            cached.remove(dataset_info.name)
            logger.info(
                f"Dataset {dataset_info.name} of project {project_id} is not up to date and will be re-downloaded."
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


def _download_project_to_cache(
    api: Api,
    project_info: ProjectInfo,
    dataset_infos: List[DatasetInfo],
    log_progress: bool = True,
    progress_cb: Callable = None,
    **kwargs,
):
    project_id = project_info.id
    project_type = project_info.type
    kwargs = _add_save_items_infos_to_kwargs(kwargs, project_type)
    cached_project_dir = _get_cache_dir(project_id)
    if len(dataset_infos) == 0:
        logger.debug("No datasets to download")
        return
    elif is_cached(project_id):
        temp_pr_dir = os.path.join(apps_cache_dir(), rand_str(10))
        download(
            api=api,
            project_id=project_id,
            dest_dir=temp_pr_dir,
            dataset_ids=[info.id for info in dataset_infos],
            log_progress=log_progress,
            progress_cb=progress_cb,
            **kwargs,
        )
        existing_project = Project(cached_project_dir, OpenMode.READ)
        for dataset in existing_project.datasets:
            dataset: Dataset
            dataset.directory
            if dataset.name in [info.name for info in dataset_infos]:
                continue
            copy_dir_recursively(dataset.directory, os.path.join(temp_pr_dir, dataset.name))
        remove_dir(cached_project_dir)
        shutil.move(temp_pr_dir, cached_project_dir)
    else:
        download(
            api=api,
            project_id=project_id,
            dest_dir=cached_project_dir,
            dataset_ids=[info.id for info in dataset_infos],
            log_progress=log_progress,
            progress_cb=progress_cb,
            **kwargs,
        )


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

    :return: Tuple where the first list contains names of downloaded datasets and the second list contains
    names of cached datasets
    :rtype: Tuple[List, List]
    """
    project_info = api.project.get_info_by_id(project_id)
    project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    if dataset_infos is not None and dataset_ids is not None:
        raise ValueError("dataset_infos and dataset_ids cannot be specified at the same time")
    if dataset_infos is None:
        if dataset_ids is None:
            dataset_infos = api.dataset.get_list(project_id)
        else:
            dataset_infos = [api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids]
    name_to_info = {info.name: info for info in dataset_infos}
    to_download, cached = _validate(api, project_info, project_meta, dataset_infos)
    if progress_cb is not None:
        cached_items_n = sum(name_to_info[ds_name].items_count for ds_name in cached)
        progress_cb(cached_items_n)
    _download_project_to_cache(
        api=api,
        project_info=project_info,
        dataset_infos=[name_to_info[name] for name in to_download],
        log_progress=log_progress,
        progress_cb=progress_cb,
        **kwargs,
    )
    return to_download, cached


def copy_from_cache(
    project_id: int, dest_dir: str, dataset_names: List[str] = None, progress_cb: Callable = None
):
    """
    Copy project or dataset from cache to the specified directory.
    If dataset_name is None, the whole project will be copied.

    :param project_id: Project ID, which will be downloaded.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_name: Name of the dataset to copy. If not specified, the whole project will be copied.
    :type dataset_name: str, optional
    :param progress_cb: Function for tracking copying progress. Will be called with number of bytes copied.
    :type progress_cb: tqdm or callable, optional

    :return: None.
    :rtype: NoneType
    """
    if not is_cached(project_id):
        raise RuntimeError(f"Project {project_id} is not cached")
    if dataset_names is not None:
        for dataset_name in dataset_names:
            if not is_cached(project_id, dataset_name):
                raise RuntimeError(f"Dataset {dataset_name} of project {project_id} is not cached")
    cache_dir = _get_cache_dir(project_id)
    if dataset_names is None:
        copy_dir_recursively(cache_dir, dest_dir, progress_cb)
    else:
        # copy meta
        copy_file(os.path.join(cache_dir, "meta.json"), os.path.join(dest_dir, "meta.json"))
        # copy datasets
        for dataset_name in dataset_names:
            copy_dir_recursively(
                os.path.join(cache_dir, dataset_name),
                os.path.join(dest_dir, dataset_name),
                progress_cb,
            )


def download_using_cache(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
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

    :return: None.
    :rtype: NoneType
    """
    downloaded, cached = download_to_cache(
        api,
        project_id,
        dataset_ids=dataset_ids,
        log_progress=log_progress,
        progress_cb=progress_cb,
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
