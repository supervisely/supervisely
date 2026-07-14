from collections import defaultdict
from typing import Dict, List, Tuple

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.project.project_meta import ProjectMeta


def create_dataset_mapping(
    src_ds_tree: Dict[DatasetInfo, Dict],
    dst_ds_tree: Dict[DatasetInfo, Dict],
) -> Tuple[Dict[int, int], List[Tuple[int, DatasetInfo]]]:
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
        - ds_to_create: List of tuples (parent_dst_id, src_ds_info) for datasets that need to be created
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


def get_diffs(api: Api, src_project_id: int, dst_project_id: int) -> Tuple:
    """
    Get the images that are different between source and destination datasets.

    Args:
        api: sly.API instance
        src_project_id: ID of the source project
        dst_project_id: ID of the destination project

    Returns:
        Tuple containing:
        - diff_images: Dictionary mapping source dataset IDs to lists of different images
        - src_to_dst_map: Dictionary mapping source dataset IDs to destination dataset IDs
        - ds_to_create: List of src dataset IDs that need to be recreated in the destination project
    """
    src_datasets = api.dataset.get_list(src_project_id, recursive=True)
    src_tree = api.dataset.get_tree(src_project_id)
    dst_tree = api.dataset.get_tree(dst_project_id)

    src_to_dst_map, ds_to_create = create_dataset_mapping(src_tree, dst_tree)

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

    return diff_images, src_to_dst_map, ds_to_create


def merge_update_metas(api: Api, src_project_id: int, dst_project_id: int):
    meta_1 = api.project.get_meta(src_project_id, with_settings=True)
    meta_2 = api.project.get_meta(dst_project_id, with_settings=True)

    meta_1 = ProjectMeta.from_json(meta_1)
    meta_2 = ProjectMeta.from_json(meta_2)

    if meta_1 != meta_2:
        meta_2 = meta_1.merge(meta_2)
        api.project.update_meta(dst_project_id, meta_2)
