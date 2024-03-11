import os
from typing import Callable, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely import get_project_class
from supervisely._utils import batched
from supervisely.annotation.annotation import Annotation, ProjectMeta
from supervisely.annotation.tag_collection import TagCollection
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.io.env import apps_cache_dir
from supervisely.io.fs import (
    clean_dir,
    copy_dir_recursively,
    copy_file,
    dir_exists,
    get_directory_size,
    mkdir,
)
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project import Project
from supervisely.project.project import OpenMode
from supervisely.task.progress import Progress


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


def _download_datasets_to_existing_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids=List[int],
    log_progress=False,
    batch_size=50,
    only_image_tags=False,
    save_image_info=False,
    save_images=True,
    progress_cb=None,
    save_image_meta=False,
):
    # if meta not found, download it
    meta_path = os.path.join(dest_dir, "meta.json")
    if not os.path.exists(meta_path):
        dump_json_file(api.project.get_meta(project_id), meta_path)
    try:
        project_fs = Project(dest_dir, OpenMode.READ)
    except RuntimeError as e:
        # if project is empty, read meta, clean dir, create project and set meta
        if str(e) == "Project is empty":
            meta_path = os.path.join(dest_dir, "meta.json")
            meta = ProjectMeta.from_json(load_json_file(meta_path))
            clean_dir(dest_dir)
            project_fs = Project(dest_dir, OpenMode.CREATE)
            project_fs.set_meta(meta)
        else:
            raise e

    for dataset_id in dataset_ids:
        dataset_info = api.dataset.get_info_by_id(dataset_id)

        dataset_fs = project_fs.create_dataset(dataset_info.name)
        images = api.image.get_list(dataset_id)

        if save_image_meta:
            meta_dir = os.path.join(dest_dir, dataset_info.name, "meta")
            mkdir(meta_dir)
            for image_info in images:
                meta_paths = os.path.join(meta_dir, image_info.name + ".json")
                dump_json_file(image_info.meta, meta_paths)

        ds_progress = None
        if log_progress:
            ds_progress = Progress(
                "Downloading dataset: {!r}".format(dataset_info.name),
                total_cnt=len(images),
            )

        for batch in batched(images, batch_size):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download images in numpy format
            if save_images:
                batch_imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
            else:
                batch_imgs_bytes = [None] * len(image_ids)

            # download annotations in json format
            if only_image_tags is False:
                ann_infos = api.annotation.download_batch(dataset_id, image_ids)
                ann_jsons = [ann_info.annotation for ann_info in ann_infos]
            else:
                id_to_tagmeta = project_fs.meta.tag_metas.get_id_mapping()
                ann_jsons = []
                for image_info in batch:
                    tags = TagCollection.from_api_response(
                        image_info.tags, project_fs.meta.tag_metas, id_to_tagmeta
                    )
                    tmp_ann = Annotation(
                        img_size=(image_info.height, image_info.width), img_tags=tags
                    )
                    ann_jsons.append(tmp_ann.to_json())

            for img_info, name, img_bytes, ann in zip(
                batch, image_names, batch_imgs_bytes, ann_jsons
            ):
                dataset_fs.add_item_raw_bytes(
                    item_name=name,
                    item_raw_bytes=img_bytes if save_images is True else None,
                    ann=ann,
                    img_info=img_info if save_image_info is True else None,
                )

            if log_progress:
                ds_progress.iters_done_report(len(batch))
            if progress_cb is not None:
                progress_cb(len(batch))


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
    if is_cached(project_id):
        _download_datasets_to_existing_project(
            api=api,
            project_id=project_id,
            dest_dir=cache_project_dir,
            dataset_ids=dataset_ids,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    else:
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
    log_progress: Optional[bool] = False,
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
    :type log_progress: bool, optional
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
