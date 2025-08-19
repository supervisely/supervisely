from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.image_api import ImageInfo
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


def compare_dataset_structure(
    src_ds_tree: Dict[DatasetInfo, Dict],
    dst_ds_tree: Dict[DatasetInfo, Dict],
) -> Tuple[Dict[int, Optional[int]], List[int]]:
    """
    Create a mapping between source and destination datasets to handle nested datasets.

    Args:
        src_ds_tree: The source project dataset tree
        dst_ds_tree: The destination project dataset tree
        src_datasets: Flat list of source datasets
        dst_datasets: Flat list of destination datasets

    Returns:
        A tuple containing:
        - src_to_dst_map: Dict mapping source dataset IDs to destination dataset IDs (or None if not exists)
        - ds_to_create: List of source dataset IDs that need to be created in the destination project
    """
    # Maps source dataset IDs to destination dataset IDs (or None if it doesn't exist)
    src_to_dst_map = {}

    # List of (parent_dst_id, src_ds_info) pairs for datasets that need to be created
    ds_to_create = []

    # Helper function to build mapping recursively
    def process_datasets(src_tree, dst_tree, parent_dst_id=None):
        for src_ds_info, src_children in src_tree.items():
            # Try to find matching dataset in destination by name
            dst_ds_info = None
            dst_children = {}

            for dst_info, dst_child in dst_tree.items():
                if dst_info.name == src_ds_info.name:
                    dst_ds_info = dst_info
                    dst_children = dst_child
                    break

            if dst_ds_info:
                # Dataset exists in destination
                src_to_dst_map[src_ds_info.id] = dst_ds_info.id
                # Process children recursively
                process_datasets(src_children, dst_children, dst_ds_info.id)
            else:
                # Dataset doesn't exist in destination, needs to be created
                src_to_dst_map[src_ds_info.id] = None
                ds_to_create.append(src_ds_info.id)
                # Process children recursively with None as the parent ID
                # (they'll be created after their parent)
                process_datasets(src_children, {}, None)

    # Start the recursive mapping
    process_datasets(src_ds_tree, dst_ds_tree)

    return src_to_dst_map, ds_to_create


def compare_projects(api: Api, src_project_id: int, dst_project_id: int) -> Dict[int, List]:
    """
    Get the images that are different between source and destination datasets.

    :param api: The API instance to interact with Supervisely.
    :type api: Api
    :param src_project_id: ID of the source project.
    :type src_project_id: int
    :param dst_project_id: ID of the destination project.
    :type dst_project_id: int
    :return: A dictionary where keys are source dataset IDs and values are lists of images that are in the source but not in the destination.
    :rtype: Dict[int, List[ImageInfo]]

    Example of returned dictionary:
    {
        123: [
            ImageInfo(id=1, name='image1.jpg', ...),
            ImageInfo(id=2, name='image2.jpg', ...)
        ],
        456: [ImageInfo(id=3, name='image3.jpg', ...)]
    }
    """
    src_datasets = api.dataset.get_list(src_project_id, recursive=True)
    src_tree = api.dataset.get_tree(src_project_id)
    dst_tree = api.dataset.get_tree(dst_project_id)

    src_to_dst_map, _ = compare_dataset_structure(src_tree, dst_tree)

    diff_images = defaultdict(list)
    for src_ds in src_datasets:
        dst_ds = src_to_dst_map.get(src_ds.id)
        src_imgs = api.image.get_list(src_ds.id, force_metadata_for_links=False)
        if dst_ds is None:
            diff_images[src_ds.id].extend(src_imgs)
        else:
            dst_imgs = api.image.get_list(dst_ds, force_metadata_for_links=False)
            src_imgs_dict = {img.name: img for img in src_imgs}
            dst_imgs_dict = {img.name: img for img in dst_imgs}
            for img_name, img in src_imgs_dict.items():
                if img_name not in dst_imgs_dict:
                    diff_images[src_ds.id].append(img)

    return diff_images


def merge_update_metas(api: Api, src_project_id: int, dst_project_id: int):
    meta_1 = api.project.get_meta(src_project_id, with_settings=True)
    meta_2 = api.project.get_meta(dst_project_id, with_settings=True)

    meta_1 = ProjectMeta.from_json(meta_1)
    meta_2 = ProjectMeta.from_json(meta_2)

    if meta_1 != meta_2:
        meta_2 = meta_1.merge(meta_2)
        api.project.update_meta(dst_project_id, meta_2)


def copy_structured_images(
    api: Api,
    src_project_id: int,
    dst_project_id: int,
    images: Dict[int, List[ImageInfo]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Copy or move images from the source project to the destination project.

    :param dst_project_id: int, ID of the destination project.
    :type dst_project_id: int
    :param images: dict, Dictionary of sampled images where keys are source dataset IDs and values are lists of ImageInfo objects.
    :type images: dict
    :return: Tuple containing two dictionaries:
        - Source dataset IDs to lists of source image IDs.
        - Destination dataset IDs to lists of destination image IDs.
    :rtype: Tuple[Dict[int, List[int]], Dict[int, List[int]]]
    """
    if isinstance(images, List):
        if all(isinstance(i, int) for i in images):
            images = api.image.get_info_by_id_batch(images, force_metadata_for_links=False)
        temp_images = defaultdict(list)
        for img in images:
            if isinstance(img, ImageInfo):
                temp_images[img.dataset_id].append(img)
            else:
                logger.warning(f"Expected ImageInfo object, got {type(img)}. Skipping this image.")
        images = temp_images

    # Prepare children-parent relationships for source and destination datasets
    src_tree = api.dataset.get_tree(src_project_id)
    dst_tree = api.dataset.get_tree(dst_project_id)
    src_to_dst_map, ds_to_create = compare_dataset_structure(src_tree, dst_tree)
    merge_update_metas(api, src_project_id, dst_project_id)
    src_datasets = api.dataset.get_list(src_project_id, recursive=True)
    src_id_to_info = {ds.id: ds for ds in src_datasets}
    src_child_to_parents = {ds.id: [] for ds in src_datasets}
    for ds in src_datasets:
        current = ds
        while parent_id := current.parent_id:
            src_child_to_parents[ds.id].append(parent_id)
            current = src_id_to_info[parent_id]

    dst = {}
    src = {}
    for src_ds_id, src_imgs in images.items():
        if len(src_imgs) > 0:
            dst_ds_id = src_to_dst_map.get(src_ds_id)
            if dst_ds_id is None and src_ds_id in ds_to_create:
                # Create new dataset in destination project
                src_parent_ids = src_child_to_parents[src_ds_id]
                dst_parent_id = None
                for parent_id in src_parent_ids:
                    src_ds = api.dataset.get_info_by_id(parent_id)
                    dst_ds = api.dataset.create(
                        dst_project_id,
                        src_ds.name,
                        parent_id=dst_parent_id,
                    )
                    dst_parent_id = dst_ds.id
                    src_to_dst_map[parent_id] = dst_parent_id

                # Create new dataset in destination project
                src_ds = api.dataset.get_info_by_id(src_ds_id)
                dst_ds = api.dataset.create(
                    dst_project_id,
                    src_ds.name,
                    parent_id=dst_parent_id,
                )
                dst_ds_id = dst_ds.id
                src_to_dst_map[src_ds_id] = dst_ds_id

            new_imgs = api.image.copy_batch_optimized(
                src_dataset_id=src_ds_id,
                src_image_infos=src_imgs,
                dst_dataset_id=dst_ds_id,
                with_annotations=True,
                save_source_date=False,
            )
            src[src_ds_id] = [i.id for i in src_imgs]
            dst[dst_ds_id] = [i.id for i in new_imgs]
            logger.info(f"Copied {len(new_imgs)} images to dataset {dst_ds_id}")
    total_copied = sum(len(v) for v in dst.values())
    return src, dst, total_copied


def move_structured_images(
    api: Api,
    src_project_id: int,
    dst_project_id: int,
    images: Dict[int, List[ImageInfo]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Move images from the source project to the destination project.

    :param api: Api instance to interact with Supervisely.
    :param src_project_id: ID of the source project.
    :param dst_project_id: ID of the destination project.
    :param images: Dictionary of sampled images where keys are source dataset IDs and values are lists of ImageInfo objects.
    :return: Tuple containing two dictionaries:
        - Source dataset IDs to lists of source image IDs.
        - Destination dataset IDs to lists of destination image IDs.
    """
    src, dst, total_moved = copy_structured_images(api, src_project_id, dst_project_id, images)
    # ? check if they are removed from source EntitiesCollection
    all_src_ids = [img_id for img_ids in src.values() for img_id in img_ids]
    api.image.remove_batch(all_src_ids)
    logger.info(f"Removed {len(all_src_ids)} images from source project {src_project_id}")
    return src, dst, total_moved

