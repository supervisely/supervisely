import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.entities_collection_api import (
    AiSearchThresholdDirection,
    CollectionTypeFilter,
)
from supervisely.api.image_api import ImageInfo
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.solutions.constants import EMBEDDINGS_GENERATOR_SLUG


class SamplingSettings(TypedDict, total=False):
    mode: str
    sample_size: NotRequired[int]
    diversity_mode: NotRequired[str]
    prompt: NotRequired[str]
    limit: NotRequired[int]


class SamplingMode(Enum):
    RANDOM = "Random"
    DIVERSE = "Diverse"
    AI_SEARCH = "AI Search"


def compare_dataset_structure(
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


def get_difference(api: Api, src_project_id: int, dst_project_id: int) -> Tuple:
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


def sample(
    api: Api,
    team_id: int,
    src_project_id: int,
    dst_project_id: int,
    sampled_images: Dict[str, List[int]],
    settings: dict,
):
    """
    Run the sampling process based on the current settings.
    """
    diffs = get_difference(api, src_project_id, dst_project_id)
    mode = settings["mode"]

    if not diffs:
        raise ValueError("No new items to copy to the labeling project.")

    # Filter out already sampled images
    filtered_diffs = {}
    for ds_id, imgs in diffs.items():
        ignore_ids = {img for img in sampled_images.get(ds_id, [])}
        filtered_diffs[ds_id] = [img for img in imgs if img.id not in ignore_ids]

    total_diffs = sum(len(imgs) for imgs in filtered_diffs.values())

    if total_diffs == 0:
        raise ValueError("No new items to copy to the labeling project.")

    # If the sample size is greater than the total differences, return all images
    sample_size = settings.get("sample_size", None)
    if sample_size and sample_size >= total_diffs:
        logger.warning(
            f"Sample size ({sample_size}) is greater than total differences ({total_diffs}). "
            "Returning all images."
        )
        return filtered_diffs

    # Calculate the sample size for each dataset
    samples_per_dataset = {}
    remaining = sample_size or total_diffs
    for ds_id, imgs in filtered_diffs.items():
        if sample_size is not None:
            ds_sample = int((len(imgs) / total_diffs) * sample_size)
        else:
            ds_sample = len(imgs)
        samples_per_dataset[ds_id] = ds_sample
        remaining -= ds_sample

    # Distribute any remaining samples randomly
    if remaining > 0:
        datasets_with_space = [
            ds_id
            for ds_id, imgs in filtered_diffs.items()
            if len(imgs) > samples_per_dataset[ds_id]
        ]
        while remaining > 0 and datasets_with_space:
            ds_id = random.choice(datasets_with_space)
            if len(filtered_diffs[ds_id]) > samples_per_dataset[ds_id]:
                samples_per_dataset[ds_id] += 1
                remaining -= 1
            else:
                datasets_with_space.remove(ds_id)

    if mode not in [
        SamplingMode.RANDOM.value,
        SamplingMode.DIVERSE.value,
        SamplingMode.AI_SEARCH.value,
    ]:
        raise ValueError(f"Unknown sampling mode: {mode}")

    if mode == SamplingMode.RANDOM.value:
        new_sampled_images = {}
        for ds_id, sample_count in samples_per_dataset.items():
            if sample_count > 0:
                new_sampled_images[ds_id] = random.sample(filtered_diffs[ds_id], sample_count)
        return new_sampled_images

    all_diffs_flat = []
    for ds_id, imgs in filtered_diffs.items():
        all_diffs_flat.extend([img.id for img in imgs])
    logger.info(f"Sample mode: {mode}. Settings: {settings}")

    method = "diverse" if mode == SamplingMode.DIVERSE.value else "search"
    data = {"project_id": src_project_id}
    data["image_ids"] = all_diffs_flat

    if mode == SamplingMode.AI_SEARCH.value:  # AI search mode
        prompt = settings.get("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required for AI search mode.")
        data["prompt"] = prompt
        data["limit"] = settings.get("limit", None)
        data["threshold"] = settings.get("threshold", 0.05)
    else:  # Diverse mode
        data["sample_size"] = sample_size
        data["num_clusters"] = sample_size
        data["clustering_method"] = "kmeans"
        data["sampling_method"] = "centroids"
    # Send request to the API
    module_info = api.app.get_ecosystem_module_info(slug=EMBEDDINGS_GENERATOR_SLUG)
    sessions = api.app.get_sessions(team_id, module_info.id, statuses=[api.task.Status.STARTED])
    if len(sessions) == 0:
        raise RuntimeError("No active sessions found for embeddings generator.")
    session = sessions[0]
    # api.app.wait(session.task_id, target_status=api.task.Status.STARTED)
    logger.info(f"Embeddings generator session: {session.task_id}")
    res = api.app.send_request(session.task_id, method, data=data)
    if isinstance(res, dict):
        if "collection_id" in res:
            collection_id = res["collection_id"]

            if mode == SamplingMode.AI_SEARCH.value:
                all_sampled_images = api.entities_collection.get_items(
                    collection_id=collection_id,
                    collection_type=CollectionTypeFilter.AI_SEARCH,
                    ai_search_threshold=data.get("threshold", 0.05),
                    ai_search_threshold_direction=AiSearchThresholdDirection.ABOVE,
                )
            else:
                all_sampled_images = api.entities_collection.get_items(
                    collection_id=collection_id,
                    collection_type=CollectionTypeFilter.DEFAULT,
                )
            new_sampled_images = {}
            for img in all_sampled_images:
                ds_id = img.dataset_id
                if ds_id not in new_sampled_images:
                    new_sampled_images[ds_id] = []
                new_sampled_images[ds_id].append(img)
        elif "message" in res:
            raise RuntimeError(f"Error during sampling: {res['message']}")
    elif isinstance(res, list):
        res_ids = {img["id"] for img in res}
        new_sampled_images = {
            ds_id: [img for img in filtered_diffs[ds_id] if img.id in res_ids]
            for ds_id in filtered_diffs.keys()
        }
    else:
        raise TypeError(f"Unexpected response type: {type(res)}. Expected dict or list.")
    return new_sampled_images


def preview_sample(
    api: Api,
    team_id: int,
    src_project_id: int,
    dst_project_id: int,
    sampled_images: Dict[str, List[int]],
    settings: dict,
) -> List[ImageInfo]:
    """
    Preview the sampled images based on the current settings.
    """
    images = sample(
        api,
        team_id,
        src_project_id,
        dst_project_id,
        sampled_images,
        settings,
    )
    preview_images = []
    for _, imgs in images.items():
        preview_images.extend(imgs)

    return random.choices(preview_images, k=6)  # Randomly select 6 images for preview


def copy_to_new_project(
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
    # Prepare children-parent relationships for source and destination datasets
    src_tree = api.dataset.get_tree(src_project_id)
    dst_tree = api.dataset.get_tree(dst_project_id)
    src_to_dst_map, ds_to_create = compare_dataset_structure(src_tree, dst_tree)
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
