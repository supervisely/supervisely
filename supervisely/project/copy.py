from __future__ import annotations

import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional
from uuid import UUID

from tqdm import tqdm

import supervisely as sly
from supervisely import logger
from supervisely.api.api import ApiField
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.project.project_settings import LabelingInterface

UPLOAD_IMAGES_BATCH_SIZE = 1000


class ConflictMode(str, Enum):
    """Conflict resolution strategy when a dataset or item with the same name
    already exists at the destination."""

    SKIP = "skip"
    RENAME = "rename"
    REPLACE = "replace"


class CreatedDataset:
    """Result of copying a single dataset.

    :param src: The source DatasetInfo (or ProjectInfo for top-level copies).
    :type src: sly.DatasetInfo or sly.ProjectInfo
    :param dst_dataset: The created destination DatasetInfo, or ``None`` when
        the dataset was skipped due to a conflict.
    :type dst_dataset: sly.DatasetInfo or None
    :param conflict_resolution_result: One of ``"copied"``, ``"renamed"``,

        ``"replaced"``, or ``"skipped"``.
    :type conflict_resolution_result: str
    """

    __slots__ = ("src", "dst_dataset", "conflict_resolution_result")

    def __init__(
        self,
        src: sly.DatasetInfo | sly.ProjectInfo,
        dst_dataset: Optional[sly.DatasetInfo],
        conflict_resolution_result: str,  # "copied" | "skipped" | "replaced" | "renamed"
    ):
        self.src = src
        self.dst_dataset = dst_dataset
        self.conflict_resolution_result = conflict_resolution_result

    def __repr__(self) -> str:
        return (
            f"<CreatedDataset src={getattr(self.src, 'name', '?')!r} "
            f"dst={getattr(self.dst_dataset, 'name', None)!r} "
            f"result={self.conflict_resolution_result!r}>"
        )


def images_get_list(
    api: sly.Api, dataset_id: int, image_ids: Optional[List[int]] = None
) -> List[sly.ImageInfo]:
    """Fetch image infos for a dataset, requesting only the fields needed for copying.

    Kept in ``copy.py`` because it uses a hand-picked subset of ``ApiField``
    values that the copy pipeline depends on.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param dataset_id: Dataset whose images to list.
    :type dataset_id: int
    :param image_ids: When provided, only these image IDs are returned.
    :type image_ids: list of int, optional
    :returns: List of ``ImageInfo`` objects.
    :rtype: list[sly.ImageInfo]
    """
    api_fields = [
        ApiField.ID,
        ApiField.NAME,
        ApiField.HASH,
        ApiField.DATASET_ID,
        ApiField.CREATED_AT,
        ApiField.UPDATED_AT,
        ApiField.META,
        ApiField.PATH_ORIGINAL,
        ApiField.CREATED_BY_ID[0][0],
        ApiField.DESCRIPTION,
    ]
    if image_ids is None:
        img_infos = api.image.get_list(
            dataset_id, fields=api_fields, force_metadata_for_links=False
        )
    else:
        img_infos = api.image.get_info_by_id_batch(
            ids=image_ids, fields=api_fields, force_metadata_for_links=False
        )
    return img_infos


def images_bulk_add(
    api: sly.Api,
    dataset_id: int,
    names: List[str],
    image_infos: List[sly.ImageInfo],
    preserve_dates: bool = False,
) -> List[sly.ImageInfo]:
    """Add a batch of images to a dataset by hash or link.

    Kept in ``copy.py`` to avoid a circular import: moving this to
    ``project.py`` would require ``project.py`` to import from ``copy.py``,
    which already imports supervisely (transitively importing ``project.py``).

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param dataset_id: Destination dataset ID.
    :type dataset_id: int
    :param names: List of names to assign to the uploaded images.
    :type names: list[str]
    :param image_infos: Source ``ImageInfo`` list.  Each entry is used to obtain
        the hash, link, meta, description, and optionally the creator /
        timestamp fields.
    :type image_infos: list[sly.ImageInfo]
    :param preserve_dates: When ``True``, copy ``created_at``, ``updated_at``,
        and ``created_by`` from the source infos.
    :type preserve_dates: bool
    :returns: List of ``ImageInfo`` objects for the newly added destination images.
    :rtype: list[sly.ImageInfo]
    :raises ValueError: If the image creator is not a member of the destination team.
    """
    img_data = []
    for name, img_info in zip(names, image_infos):
        img_json = {
            ApiField.NAME: name,
            ApiField.META: img_info.meta,
        }
        if preserve_dates:
            img_json[ApiField.CREATED_AT] = img_info.created_at
            img_json[ApiField.UPDATED_AT] = img_info.updated_at
            img_json[ApiField.CREATED_BY_ID[0][0]] = img_info.created_by
        if img_info.link is not None:
            img_json[ApiField.LINK] = img_info.link
        elif img_info.hash is not None:
            img_json[ApiField.HASH] = img_info.hash
        if img_info.description is not None:
            img_json[ApiField.DESCRIPTION] = img_info.description
        img_data.append(img_json)

    try:
        response = api.post(
            "images.bulk.add",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.IMAGES: img_data,
                ApiField.FORCE_METADATA_FOR_LINKS: False,
                ApiField.SKIP_VALIDATION: True,
            },
        )
    except Exception as e:
        if "Some users are not members of the destination group" in str(e):
            raise ValueError(
                "Unable to add images. Image creator is not a member of the destination team."
            ) from e
        else:
            raise e

    results = []
    for info_json in response.json():
        info_json_copy = info_json.copy()
        if info_json.get(ApiField.MIME, None) is not None:
            info_json_copy[ApiField.EXT] = info_json[ApiField.MIME].split("/")[1]
        results.append(api.image._convert_json_info(info_json_copy))
    return results


def _create_one_dataset(
    api: sly.Api,
    dataset_info: sly.DatasetInfo,
    dst_project_id: int,
    dst_parent_id: Optional[int],
    project_type: str,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    preserve_dates: bool = False,
    clone_annotations: bool = True,
    dst_existing_names: Optional[set] = None,
    progress_cb: Optional[Callable] = None,
) -> CreatedDataset:
    """Create one dataset in the destination project and clone its items.

    Conflict resolution logic is fully encapsulated here.  The caller provides
    ``dst_existing_names`` (the set of dataset names already present at the
    same hierarchy level in the destination project) which is obtained via a
    single ``api.dataset.get_list`` call per level, eliminating N redundant
    requests.

    Improvements over the original implementation:

    * Removed the redundant ``api.dataset.get_info_by_id`` at the top
      (``custom_data`` is now pre-fetched in a batch by the caller).
    * Removed the ``api.dataset.get_info_by_id`` at the end that was used
      only to refresh ``items_count``.
    * Removed the extra ``api.dataset.update(custom_data=...)`` call because
      ``custom_data`` is now passed directly to ``create_dataset``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param dataset_info: Full ``DatasetInfo`` of the source dataset (including
        ``custom_data``).
    :type dataset_info: sly.DatasetInfo
    :param dst_project_id: ID of the destination project.
    :type dst_project_id: int
    :param dst_parent_id: ID of the parent dataset in the destination project, or
        ``None`` for top-level datasets.
    :type dst_parent_id: int or None
    :param project_type: Project type string (``sly.ProjectType.*``).
    :type project_type: str
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param options: Dict with conflict-resolution and clone flags.
    :type options: dict
    :param dst_existing_names: Names of datasets already present at this level in
        the destination; used for conflict detection.  Pass ``None`` to
        skip conflict checks.
    :type dst_existing_names: set, optional
    :param progress_cb: Optional callable ``(n_items: int)`` for progress updates.
    :type progress_cb: callable, optional
    :returns: A :class:`CreatedDataset` describing the outcome.
    :rtype: CreatedDataset
    """

    # ── SKIP: check before creation ──────────────────────────────────────────
    if conflict_mode == ConflictMode.SKIP and dst_existing_names is not None:
        if dataset_info.name in dst_existing_names:
            logger.info(
                "Dataset already exists at destination, skipping",
                extra={"dataset_name": dataset_info.name, "dst_parent_id": dst_parent_id},
            )
            if progress_cb is not None:
                progress_cb(dataset_info.items_count)
            return CreatedDataset(dataset_info, None, conflict_resolution_result="skipped")

    # ── Create the dataset ──────────────────────────────────────────────────────────
    created_info: sly.DatasetInfo = api.dataset.create(
        project_id=dst_project_id,
        name=dataset_info.name,
        description=dataset_info.description,
        custom_data=dataset_info.custom_data,  # pass upfront — no second update needed
        change_name_if_conflict=True,
        parent_id=dst_parent_id,
        created_at=dataset_info.created_at if preserve_dates else None,
        updated_at=dataset_info.updated_at if preserve_dates else None,
        created_by=dataset_info.created_by if preserve_dates else None,
    )

    # ── Clone items ───────────────────────────────────────────────────────────
    copy_items(
        api=api,
        src_dataset_id=dataset_info.id,
        dst_dataset_id=created_info.id,
        project_type=project_type,
        project_meta=project_meta,
        conflict_mode=conflict_mode,
        preserve_dates=preserve_dates,
        clone_annotations=clone_annotations,
        progress_cb=progress_cb,
    )

    # ── REPLACE: rename / remove the old dataset ─────────────────────────────
    if conflict_mode == ConflictMode.REPLACE and dst_existing_names is not None:
        if dataset_info.name in dst_existing_names:
            existing_info = api.dataset.get_info_by_name(
                dst_project_id, name=dataset_info.name, parent_id=dst_parent_id
            )
            if existing_info is not None and existing_info.id != created_info.id:
                created_info = _replace_dataset(api, existing_info, created_info)
                return CreatedDataset(
                    dataset_info, created_info, conflict_resolution_result="replaced"
                )

    # ── Determine the final resolution label ─────────────────────────────────
    if created_info.name != dataset_info.name:
        result = "renamed"
    else:
        result = "copied"

    return CreatedDataset(dataset_info, created_info, conflict_resolution_result=result)


def create_dataset_tree(
    api: sly.Api,
    datasets_tree: Dict[sly.DatasetInfo, Dict],
    dst_project_id: int,
    dst_parent_id: Optional[int],
    project_type: str,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    preserve_dates: bool = False,
    clone_annotations: bool = True,
    progress_cb: Optional[Callable] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> List[CreatedDataset]:
    """Recursively (iteratively via BFS) create a tree of datasets.

    Improvements over the original ``create_dataset_recursively``:

    1. Full dataset infos (including ``custom_data``) are pre-loaded in a
       single batch before processing starts, not one-by-one inside recursion.
    2. The list of datasets at each destination level is cached — one
       ``api.dataset.get_list`` call per level instead of N calls.
    3. True level-wise parallelism via a futures list per tree level;
       replaces the Queue + ``_consume()`` artifact.
    4. No extra ``api.dataset.get_info_by_id`` or ``api.dataset.update`` calls.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param datasets_tree: Nested dict ``{DatasetInfo: {children...}}`` describing
        the source dataset tree.
    :type datasets_tree: dict[sly.DatasetInfo, dict]
    :param dst_project_id: ID of the destination project.
    :type dst_project_id: int
    :param dst_parent_id: ID of the parent dataset for the tree root, or ``None``
        for top-level datasets.
    :type dst_parent_id: int or None
    :param project_type: Project type string.
    :type project_type: str
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode.
    :type conflict_mode: ConflictMode
    :param preserve_dates: When ``True``, copies ``created_at``, ``updated_at``,
        and ``created_by`` from the source datasets.
    :type preserve_dates: bool
    :param clone_annotations: When ``True``, annotations are cloned.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)`` for progress tracking.
    :type progress_cb: callable, optional
    :param executor: Optional external ``ThreadPoolExecutor``.  If ``None``, a
        temporary executor is created internally.
    :type executor: ThreadPoolExecutor, optional
    :returns: List of :class:`CreatedDataset` objects in BFS order.
    :rtype: list[CreatedDataset]
    """
    own_executor = executor is None
    if own_executor:
        executor = ThreadPoolExecutor(max_workers=8)

    results: List[CreatedDataset] = []

    try:
        # Pre-load full dataset infos for all datasets in a single batch
        all_dataset_ids = [ds.id for ds in _flatten_tree(datasets_tree)]
        full_infos: Dict[int, sly.DatasetInfo] = _batch_get_dataset_infos(api, all_dataset_ids)

        # Cache: dst_parent_id -> set of existing dataset names
        dst_existing_cache: Dict[Optional[int], set] = {}

        def _get_dst_existing_names(parent_id: Optional[int]) -> set:
            if parent_id not in dst_existing_cache:
                existing = api.dataset.get_list(dst_project_id, parent_id=parent_id)
                dst_existing_cache[parent_id] = {ds.name for ds in existing}
            return dst_existing_cache[parent_id]

        # BFS queue: list of (subset_tree, parent_id) pairs
        queue: List[tuple] = [(datasets_tree, dst_parent_id)]

        while queue:
            level_items = queue
            queue = []

            # Process all datasets at the current level in parallel
            level_futures = {}
            for subset, parent_id in level_items:
                dst_existing = _get_dst_existing_names(parent_id)
                for ds_info, children in subset.items():
                    full_info = full_infos.get(ds_info.id, ds_info)
                    fut = executor.submit(
                        _create_one_dataset,
                        api,
                        full_info,
                        dst_project_id=dst_project_id,
                        dst_parent_id=parent_id,
                        project_type=project_type,
                        project_meta=project_meta,
                        conflict_mode=conflict_mode,
                        preserve_dates=preserve_dates,
                        clone_annotations=clone_annotations,
                        dst_existing_names=dst_existing,
                        progress_cb=progress_cb,
                    )
                    level_futures[fut] = (children, full_info.name)

            for fut in as_completed(level_futures):
                children, _ = level_futures[fut]
                created = fut.result()
                results.append(created)
                # If the dataset was created and has children, enqueue them
                if children and created.dst_dataset is not None:
                    new_parent_id = created.dst_dataset.id
                    # Evict the cache entry for the new parent (it was empty, but be explicit)
                    dst_existing_cache.pop(new_parent_id, None)
                    queue.append((children, new_parent_id))

    finally:
        if own_executor:
            executor.shutdown(wait=True)

    return results


def copy_project(
    api: sly.Api,
    src_project_info: sly.ProjectInfo,
    dst_workspace_id: int,
    dst_project_id: Optional[int] = None,
    dst_dataset_id: Optional[int] = None,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    preserve_dates: bool = False,
    clone_annotations: bool = True,
    progress_cb: Optional[Callable] = None,
    existing_projects: Optional[List[sly.ProjectInfo]] = None,
    datasets_tree: Optional[Dict] = None,
    dst_project_name: Optional[str] = None,
    dst_project_description: Optional[str] = None,
    read_only: bool = False,
) -> sly.ProjectInfo:
    """Copy a project into a workspace or into an existing project/dataset.

    Unifies the logic of the legacy ``copy_project``,
    ``copy_project_with_replace``, and ``copy_project_with_skip`` into a single
    function, eliminating code duplication.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_project_info: ``ProjectInfo`` of the project to copy.
    :type src_project_info: sly.ProjectInfo
    :param dst_workspace_id: Destination workspace ID.
    :type dst_workspace_id: int
    :param dst_project_id: Destination project ID when copying *into* an existing
        project.  Currently ``None`` is required (copying into an existing
        project is not yet implemented).
    :type dst_project_id: int or None
    :param dst_dataset_id: Destination dataset ID (reserved for future use).
    :type dst_dataset_id: int or None
    :param conflict_mode: How to handle name conflicts — ``"rename"`` (default),
        ``"skip"``, or ``"replace"``.
    :type conflict_mode: str
    :param preserve_dates: When ``True``, copy ``created_at``, ``updated_at``,
        and ``created_by`` from source objects.
    :type preserve_dates: bool
    :param clone_annotations: When ``True``, annotations are copied together
        with items.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)`` for progress reporting.
    :type progress_cb: callable, optional
    :param existing_projects: Optional pre-fetched list of projects in
        ``dst_workspace_id``.  Loaded lazily when ``None``.
    :type existing_projects: list[sly.ProjectInfo], optional
    :param datasets_tree: Optional pre-fetched source dataset tree.  Fetched once
        from the API when ``None``.
    :type datasets_tree: dict, optional
    :param dst_project_name: Optional custom name for the destination project.
        By default, the source project name is used (with automatic renaming if needed based on ``conflict_mode``).
    :type dst_project_name: str, optional
    :param dst_project_description: Optional custom description for the destination project.
        By default, the source project description is used.
    :type dst_project_description: str, optional
    :param read_only: Optional flag to set the project as read-only. Works only with image and video projects. If set to True, the project will be created with read-only settings, and users will not be able to modify annotations in this project. Default is False.
    :type read_only: bool, optional
    :returns: List of :class:`CreatedDataset` objects, one per created dataset.
        Empty list when the project was skipped.
    :rtype: list[CreatedDataset]
    :raises NotImplementedError: If ``dst_project_id`` is not ``None``.
    """

    project_type: str = src_project_info.type

    if datasets_tree is None:
        datasets_tree = api.dataset.get_tree(src_project_info.id)

    created_datasets: List[CreatedDataset] = []

    # ════════════════════════════════════════════════════════════════════════
    # Case A: copy INTO an existing project or dataset
    # TODO: implement and test when needed
    # ════════════════════════════════════════════════════════════════════════
    if dst_project_id is not None or dst_dataset_id is not None:
        raise NotImplementedError(
            "Copying into an existing project/dataset is not yet implemented. "
            "Pass dst_project_id=None and dst_dataset_id=None to copy into a workspace."
        )

    # ════════════════════════════════════════════════════════════════════════
    # Case B: copy into a workspace (creates a new project)
    # ════════════════════════════════════════════════════════════════════════

    # Lazy-load existing projects for the conflict check
    if existing_projects is None:
        existing_projects = api.project.get_list(dst_workspace_id)
    existing_by_name = {p.name: p for p in existing_projects}

    # ── SKIP ─────────────────────────────────────────────────────────────
    if conflict_mode == ConflictMode.SKIP and src_project_info.name in existing_by_name:
        logger.info(
            "Project with the same name already exists at destination. Skipping.",
            extra={"src_project_name": src_project_info.name},
        )
        if progress_cb is not None:
            progress_cb(src_project_info.items_count)
        return []

    # Create the new project
    created_project = api.project.create(
        workspace_id=dst_workspace_id,
        name=src_project_info.name if dst_project_name is None else dst_project_name,
        type=src_project_info.type,
        description=(
            src_project_info.description
            if dst_project_description is None
            else dst_project_description
        ),
        settings=src_project_info.settings,
        custom_data=src_project_info.custom_data,
        readme=src_project_info.readme,
        change_name_if_conflict=True,
        created_at=src_project_info.created_at if preserve_dates else None,
        updated_at=src_project_info.updated_at if preserve_dates else None,
        created_by=src_project_info.created_by_id if preserve_dates else None,
    )

    # Copy the project meta
    project_meta = sly.ProjectMeta.from_json(
        api.project.get_meta(src_project_info.id, with_settings=True)
    )
    api.project.update_meta(created_project.id, project_meta)

    # Recursively create datasets
    created_datasets.extend(
        create_dataset_tree(
            api,
            datasets_tree,
            dst_project_id=created_project.id,
            dst_parent_id=None,
            project_type=project_type,
            project_meta=project_meta,
            conflict_mode=conflict_mode,
            preserve_dates=preserve_dates,
            clone_annotations=clone_annotations,
            progress_cb=progress_cb,
        )
    )

    # ── REPLACE: swap old project out after copying ──────────────────────────
    if conflict_mode == ConflictMode.REPLACE and src_project_info.name in existing_by_name:
        old_project = existing_by_name[src_project_info.name]
        logger.info(
            "Replacing existing project",
            extra={
                "old_project_id": old_project.id,
                "new_project_id": created_project.id,
            },
        )
        created_project = _replace_project(api, old_project, created_project)

    if read_only and project_type in [sly.ProjectType.IMAGES.value, sly.ProjectType.VIDEOS.value]:
        api.project.set_read_only(
            created_project.id,
            enable=True,
        )
    return created_project


def merge_project_meta(api: sly.Api, src_project_id: int, dst_project_id: int) -> sly.ProjectMeta:
    """Merge the meta of two projects and return the updated destination meta.

    Classes that exist in the source but not in the destination are added.
    When the same class name exists on both sides but with different geometry
    types, the destination geometry type is upgraded to ``AnyGeometry``.
    Tags with conflicting ``value_type`` are skipped with a warning.
    Project settings (multiview, labeling interface) are always taken from
    the source.

    Improvements over the original:

    * ``api`` is passed explicitly — no dependency on global state.
    * Detailed comments added throughout the merge logic.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_project_id: ID of the source project.
    :type src_project_id: int
    :param dst_project_id: ID of the destination project.
    :type dst_project_id: int
    :returns: The current (and possibly updated) ``ProjectMeta`` of the
        destination project.
    :rtype: sly.ProjectMeta
    """
    src_meta = sly.ProjectMeta.from_json(api.project.get_meta(src_project_id, True))
    if src_project_id == dst_project_id:
        return src_meta

    dst_meta = sly.ProjectMeta.from_json(api.project.get_meta(dst_project_id, True))
    changed = False

    # ── Object classes ──────────────────────────────────────────────────────────────
    for obj_class in src_meta.obj_classes:
        dst_class: Optional[sly.ObjClass] = dst_meta.obj_classes.get(obj_class.name)
        if dst_class is None:
            dst_meta = dst_meta.add_obj_class(obj_class)
            changed = True
        elif (
            dst_class.geometry_type != obj_class.geometry_type
            and dst_class.geometry_type != sly.AnyGeometry
        ):
            # Geometry type conflict: upgrade to AnyGeometry
            dst_meta = dst_meta.delete_obj_class(obj_class.name)
            dst_meta = dst_meta.add_obj_class(obj_class.clone(geometry_type=sly.AnyGeometry))
            changed = True
            logger.warning(
                "ObjClass geometry type conflict, upgrading to AnyGeometry",
                extra={"class_name": obj_class.name},
            )

    # ── Tag metas ───────────────────────────────────────────────────────────────────
    for tag_meta in src_meta.tag_metas:
        dst_tag = dst_meta.get_tag_meta(tag_meta.name)
        if dst_tag is None:
            dst_meta = dst_meta.add_tag_meta(tag_meta)
            changed = True
        elif dst_tag.value_type != tag_meta.value_type:
            logger.warning(
                "Tag value_type conflict, skipping tag merge",
                extra={"tag_name": tag_meta.name},
            )

    # ── Project settings ──────────────────────────────────────────────────────────
    if src_meta.project_settings != dst_meta.project_settings:
        new_settings = dst_meta.project_settings.clone(
            multiview_enabled=src_meta.project_settings.multiview_enabled,
            multiview_tag_name=src_meta.project_settings.multiview_tag_name,
            multiview_tag_id=src_meta.project_settings.multiview_tag_id,
            multiview_is_synced=src_meta.project_settings.multiview_is_synced,
            labeling_interface=src_meta.project_settings.labeling_interface,
        )
        dst_meta = dst_meta.clone(project_settings=new_settings)
        changed = True
        logger.info("Updated project settings in destination project to match source")

    if changed:
        return api.project.update_meta(dst_project_id, dst_meta)
    return dst_meta


def copy_items(
    api: sly.Api,
    src_dataset_id: int,
    dst_dataset_id: int,
    project_type: str,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    preserve_dates: bool = False,
    clone_annotations: bool = True,
    progress_cb: Optional[Callable] = None,
    src_infos: Optional[List] = None,
) -> List:
    """Copy all items from one dataset to another.

    Dispatches to the appropriate type-specific copy function based on
    ``project_type``.  Fetches source item infos if not provided.

    Improvements over the original:

    * ``api`` is passed explicitly — no dependency on global state.
    * Removed the global ``executor`` variable; each sub-function manages its
      own thread pools locally.
    * Dispatch by ``project_type`` is expressed as a plain dict for clarity.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_dataset_id: ID of the source dataset.
    :type src_dataset_id: int
    :param dst_dataset_id: ID of the destination dataset.
    :type dst_dataset_id: int
    :param project_type: One of ``sly.ProjectType`` string values.
    :type project_type: str
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param preserve_dates: Whether to preserve original timestamps.
    :type preserve_dates: bool
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)`` for progress reporting.
    :type progress_cb: callable, optional
    :param src_infos: Optional pre-fetched list of source item infos.  Fetched
        automatically when ``None``.
    :type src_infos: list, optional
    :returns: List of destination item info objects.
    :rtype: list
    :raises ValueError: If ``project_type`` is not supported.
    """
    dispatch = {
        str(sly.ProjectType.IMAGES): _copy_image_items,
        str(sly.ProjectType.VIDEOS): _copy_video_items,
        str(sly.ProjectType.VOLUMES): _copy_volume_items,
        str(sly.ProjectType.POINT_CLOUDS): _copy_pointcloud_items,
        str(sly.ProjectType.POINT_CLOUD_EPISODES): _copy_pointcloud_episode_items,
    }
    clone_f = dispatch.get(project_type)
    if clone_f is None:
        raise ValueError(f"Unsupported project type: {project_type!r}")

    if src_infos is None:
        logger.info(
            "Fetching source item infos (this may take a while for large datasets)",
            extra={"src_dataset_id": src_dataset_id, "project_type": project_type},
        )
        src_infos = _get_item_infos(api, src_dataset_id, project_type=project_type)
        logger.info(
            "Fetched %d source item infos",
            len(src_infos),
            extra={"src_dataset_id": src_dataset_id},
        )

    dst_infos = clone_f(
        api=api,
        src_infos=src_infos,
        dst_dataset_id=dst_dataset_id,
        project_meta=project_meta,
        conflict_mode=conflict_mode,
        preserve_dates=preserve_dates,
        clone_annotations=clone_annotations,
        progress_cb=progress_cb,
    )

    logger.info(
        "Cloned %d items",
        len(dst_infos),
        extra={"src_dataset_id": src_dataset_id, "dst_dataset_id": dst_dataset_id},
    )
    return dst_infos


def _batch_get_dataset_infos(api: sly.Api, dataset_ids: List[int]) -> Dict[int, sly.DatasetInfo]:
    """Load full DatasetInfo objects for a list of IDs in parallel.

    Replaces N sequential ``api.dataset.get_info_by_id`` calls with a
    concurrent thread-pool fetch.  When the SDK gains a native
    ``api.dataset.get_info_by_ids`` batch endpoint, this function should
    delegate to it directly.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param dataset_ids: List of dataset IDs to fetch.
    :type dataset_ids: list[int]
    :returns: Dict mapping each successfully fetched dataset ID to its
        ``DatasetInfo``.
    :rtype: dict[int, sly.DatasetInfo]
    """
    if not dataset_ids:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(api.dataset.get_info_by_id, ds_id): ds_id for ds_id in dataset_ids}
        result: Dict[int, sly.DatasetInfo] = {}
        for fut in as_completed(futures):
            ds_id = futures[fut]
            try:
                info = fut.result()
                if info is not None:
                    result[ds_id] = info
            except Exception as e:
                logger.warning(
                    "Failed to get dataset info", extra={"dataset_id": ds_id, "error": str(e)}
                )
    return result


def _flatten_tree(tree: Dict) -> List[sly.DatasetInfo]:
    """DFS traversal of a dataset tree; returns a flat list of all DatasetInfo objects.

    :param tree: Nested dict ``{DatasetInfo: {children...}}``.
    :type tree: dict
    :returns: Flat list of all :class:`sly.DatasetInfo` objects in DFS order.
    :rtype: list[sly.DatasetInfo]
    """
    result = []

    def _dfs(t: Dict):
        for ds, children in t.items():
            result.append(ds)
            _dfs(children)

    _dfs(tree)
    return result


def _replace_project(
    api: sly.Api,
    old_project: sly.ProjectInfo,
    new_project: sly.ProjectInfo,
) -> sly.ProjectInfo:
    """Delete ``old_project`` and rename ``new_project`` to ``old_project.name``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param old_project: The project to remove.
    :type old_project: sly.ProjectInfo
    :param new_project: The newly created project that should take the old name.
    :type new_project: sly.ProjectInfo
    :returns: Updated ``ProjectInfo`` for ``new_project`` with the old name.
    :rtype: sly.ProjectInfo
    """
    api.project.update(old_project.id, name=old_project.name + "__to_remove")
    api.project.remove(old_project.id)
    return api.project.update(new_project.id, name=old_project.name)


def _replace_dataset(
    api: sly.Api,
    old_dataset: sly.DatasetInfo,
    new_dataset: sly.DatasetInfo,
) -> sly.DatasetInfo:
    """Delete ``old_dataset`` and rename ``new_dataset`` to ``old_dataset.name``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param old_dataset: The dataset to remove.
    :type old_dataset: sly.DatasetInfo
    :param new_dataset: The newly created dataset that should take the old name.
    :type new_dataset: sly.DatasetInfo
    :returns: Updated ``DatasetInfo`` for ``new_dataset`` with the old name.
    :rtype: sly.DatasetInfo
    """
    api.dataset.update(old_dataset.id, name=old_dataset.name + "__to_remove")
    api.dataset.remove(old_dataset.id)
    return api.dataset.update(
        new_dataset.id,
        name=old_dataset.name,
        custom_data=old_dataset.custom_data,
    )


def _get_item_infos(
    api: sly.Api,
    dataset_id: int,
    item_ids: Optional[List[int]] = None,
    project_type: str = "images",  # sly.ProjectType.IMAGES — literal used to avoid circular-import at load time
) -> List:
    """Return item infos for a dataset, optionally filtering by ``item_ids``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param dataset_id: Dataset whose items to list.
    :type dataset_id: int
    :param item_ids: When provided, only these item IDs are returned.  Currently
        only supported for images.
    :type item_ids: list[int], optional
    :param project_type: Project type string used to select the correct API method.
    :type project_type: str
    :returns: List of item info objects (type depends on ``project_type``).
    :rtype: list
    :raises ValueError: If ``project_type`` is not supported.
    """
    dispatch = {
        str(sly.ProjectType.IMAGES): lambda: images_get_list(api, dataset_id, item_ids),
        str(sly.ProjectType.VIDEOS): lambda: api.video.get_list(dataset_id),
        str(sly.ProjectType.VOLUMES): lambda: api.volume.get_list(dataset_id),
        str(sly.ProjectType.POINT_CLOUDS): lambda: api.pointcloud.get_list(dataset_id),
        str(sly.ProjectType.POINT_CLOUD_EPISODES): lambda: api.pointcloud_episode.get_list(
            dataset_id
        ),
    }
    getter = dispatch.get(project_type)
    if getter is None:
        raise ValueError(f"Unsupported project type: {project_type!r}")
    return getter()


def _rename_image(api: sly.Api, info: sly.ImageInfo, new_name: str, to_remove_info: sly.ImageInfo):
    """Free ``new_name`` by renaming ``to_remove_info``, then rename ``info`` to ``new_name``.

    Used in REPLACE mode to atomically swap an old image for a newly uploaded
    one without a name collision.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param info: The newly uploaded image that should receive ``new_name``.
    :type info: sly.ImageInfo
    :param new_name: The desired final name for ``info``.
    :type new_name: str
    :param to_remove_info: The existing image currently occupying ``new_name``.
    :type to_remove_info: sly.ImageInfo
    """
    api.image.rename(to_remove_info.id, to_remove_info.name + "__to_remove")
    api.image.rename(info.id, new_name)


def _finalize_image_replace(
    api: sly.Api,
    src: List[sly.ImageInfo],
    dst: List[sly.ImageInfo],
    to_rename: Dict[str, str],
    existing: Dict[str, sly.ImageInfo],
):
    """After upload: rename new images to their original names and remove the old ones.

    Only performs work when ``to_rename`` is non-empty (REPLACE mode).

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src: Source ``ImageInfo`` list (returned unchanged).
    :type src: list[sly.ImageInfo]
    :param dst: Destination ``ImageInfo`` list (newly uploaded images).
    :type dst: list[sly.ImageInfo]
    :param to_rename: Mapping from *temporary* name → *original* name for images
        that were uploaded with a disambiguating suffix.
    :type to_rename: dict[str, str]
    :param existing: Mapping from name → ``ImageInfo`` for images already present
        in the destination dataset before the copy started.
    :type existing: dict[str, sly.ImageInfo]
    :returns: Tuple ``(src, dst)`` — returned unchanged for chaining.
    :rtype: tuple
    """
    if not to_rename:
        return src, dst
    local_executor = ThreadPoolExecutor(max_workers=5)
    to_remove = []
    rename_tasks = []
    for dst_image_info in dst:
        if dst_image_info.name in to_rename:
            original_name = to_rename[dst_image_info.name]
            if original_name in existing:
                to_remove.append(existing[original_name])
                rename_tasks.append(
                    local_executor.submit(
                        _rename_image, api, dst_image_info, original_name, existing[original_name]
                    )
                )
    for task in as_completed(rename_tasks):
        task.result()
    local_executor.shutdown(wait=True)
    if to_remove:
        api.image.remove_batch([info.id for info in to_remove], batch_size=len(to_remove))
    return src, dst


def _copy_image_items(
    api: sly.Api,
    src_infos: List[sly.ImageInfo],
    dst_dataset_id: int,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    preserve_dates: bool = False,
    clone_annotations: bool = True,
    progress_cb=None,
    **kwargs,
) -> List[sly.ImageInfo]:
    """Copy images and their annotations from ``src_infos`` into ``dst_dataset_id``.

    Handles SKIP, RENAME, and REPLACE conflict modes.  Image uploads and
    annotation copies are parallelised with independent thread pools.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_infos: Source ``ImageInfo`` list.
    :type src_infos: list[sly.ImageInfo]
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param preserve_dates: Whether to preserve original timestamps.
    :type preserve_dates: bool
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)``.
    :type progress_cb: callable, optional
    :returns: List of destination ``ImageInfo`` objects in the same order as
        ``src_infos``.
    :rtype: list[sly.ImageInfo]
    """

    existing = {info.name: info for info in api.image.get_list(dst_dataset_id)}
    logger.info(
        "Starting image copy: %d source images, %d already in destination",
        len(src_infos),
        len(existing),
        extra={"dst_dataset_id": dst_dataset_id},
    )

    # Deduplicate by name within src (SKIP / REPLACE modes)
    if conflict_mode in (ConflictMode.SKIP, ConflictMode.REPLACE):
        seen = set()
        non_dup = []
        for img in src_infos:
            if img.name not in seen:
                non_dup.append(img)
                seen.add(img.name)
        src_infos = non_dup

    if conflict_mode == ConflictMode.SKIP:
        len_before = len(src_infos)
        src_infos = [img for img in src_infos if img.name not in existing]
        if progress_cb is not None:
            progress_cb(len_before - len(src_infos))

    if not src_infos:
        return []

    local_executor = ThreadPoolExecutor(max_workers=5)

    def _copy_imgs(names, infos):
        uploaded = images_bulk_add(
            api,
            dst_dataset_id,
            names,
            infos,
            preserve_dates=preserve_dates,
        )
        return infos, uploaded

    def _copy_anns(src: List[sly.ImageInfo], dst: List[sly.ImageInfo]):
        by_dataset = defaultdict(list)
        for s, d in zip(src, dst):
            by_dataset[s.dataset_id].append((s, d))
        for pairs in by_dataset.values():
            src_ids = [p[0].id for p in pairs]
            dst_ids = [p[1].id for p in pairs]
            try:
                api.annotation.copy_batch_by_ids(src_ids, dst_ids, save_source_date=preserve_dates)
            except Exception as e:
                if "Some users are not members of the destination group" in str(e):
                    raise ValueError(
                        "Unable to copy annotations. Annotation creator is not a member of the destination team."
                    ) from e
                raise
        return src, dst

    reserved_names = set(existing.keys())
    to_rename: Dict[str, str] = {}  # temporary_name → original_name
    upload_tasks = []
    for batch in sly.batched(src_infos, UPLOAD_IMAGES_BATCH_SIZE):
        names = [img.name for img in batch]
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if conflict_mode in (ConflictMode.RENAME, ConflictMode.REPLACE):
            for i, name in enumerate(names):
                if name in reserved_names:
                    stem, ext = os.path.splitext(name)
                    j = 0
                    new_name = name
                    while new_name in reserved_names:
                        suffix = f"_{now}" if j == 0 else f"_{now}_{j}"
                        new_name = f"{stem}{suffix}{ext}"
                        j += 1
                    names[i] = new_name
                    if conflict_mode == ConflictMode.REPLACE:
                        to_rename[new_name] = name
                    reserved_names.add(new_name)
        upload_tasks.append(local_executor.submit(_copy_imgs, names, batch))

    replace_tasks = []
    src_id_to_dst: Dict[int, sly.ImageInfo] = {}
    ann_executor = ThreadPoolExecutor(max_workers=5)
    for task in as_completed(upload_tasks):
        src_batch, dst_batch = task.result()
        for s, d in zip(src_batch, dst_batch):
            src_id_to_dst[s.id] = d
        if clone_annotations:
            ann_tasks = [ann_executor.submit(_copy_anns, src_batch, dst_batch)]
            for ann_task in as_completed(ann_tasks):
                sb, db = ann_task.result()
                replace_tasks.append(
                    local_executor.submit(_finalize_image_replace, api, sb, db, to_rename, existing)
                )
        else:
            replace_tasks.append(
                local_executor.submit(
                    _finalize_image_replace, api, src_batch, dst_batch, to_rename, existing
                )
            )
        if progress_cb is not None:
            progress_cb(len(src_batch))

    for task in as_completed(replace_tasks):
        task.result()

    local_executor.shutdown(wait=True)
    ann_executor.shutdown(wait=True)
    return [src_id_to_dst[img.id] for img in src_infos]


def _copy_video_items(
    api: sly.Api,
    src_infos: List,
    dst_dataset_id: int,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    clone_annotations: bool = True,
    progress_cb=None,
    **kwargs,
) -> List:
    """Copy videos and their annotations from ``src_infos`` into ``dst_dataset_id``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_infos: Source ``VideoInfo`` list.
    :type src_infos: list
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param project_meta: ``ProjectMeta`` of the source project (used to
        deserialise video annotations and detect multi-view mode).
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)``.
    :type progress_cb: callable, optional
    :returns: List of destination ``VideoInfo`` objects in the same order as
        ``src_infos``.
    :rtype: list
    """
    if not src_infos:
        return []

    src_dataset_id = src_infos[0].dataset_id

    existing = {info.name: info for info in api.video.get_list(dst_dataset_id)}

    if conflict_mode == ConflictMode.SKIP:
        len_before = len(src_infos)
        src_infos = [v for v in src_infos if v.name not in existing]
        if progress_cb is not None:
            progress_cb(len_before - len(src_infos))

    if not src_infos:
        return []

    local_executor = ThreadPoolExecutor(max_workers=5)

    def _copy_videos(names, ids, metas, infos):
        uploaded = api.video.upload_ids(
            dst_dataset_id, names=names, ids=ids, metas=metas, infos=infos
        )
        return infos, uploaded

    def _copy_anns(src, dst):
        anns_jsons = api.video.annotation.download_bulk(src_dataset_id, [info.id for info in src])
        dst_ids = [info.id for info in dst]
        tasks = []
        if project_meta.labeling_interface == LabelingInterface.MULTIVIEW:
            anns = []
            key_id_map = sly.KeyIdMap()
            for ann_json in anns_jsons:
                anns.append(sly.VideoAnnotation.from_json(ann_json, project_meta, key_id_map))
            tasks.append(
                local_executor.submit(api.video.annotation.upload_anns_multiview, dst_ids, anns)
            )
        else:
            for ann_json, dst_id in zip(anns_jsons, dst_ids):
                key_id_map = sly.KeyIdMap()
                ann = sly.VideoAnnotation.from_json(ann_json, project_meta, key_id_map)
                tasks.append(
                    local_executor.submit(api.video.annotation.append, dst_id, ann, key_id_map)
                )
        for t in as_completed(tasks):
            t.result()
        return src, dst

    to_rename: Dict[str, str] = {}
    reserved_names = set(existing.keys())
    upload_tasks = []
    for batch in sly.batched(src_infos):
        names = [v.name for v in batch]
        ids = [v.id for v in batch]
        metas = [v.meta for v in batch]
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if conflict_mode in (ConflictMode.RENAME, ConflictMode.REPLACE):
            for i, name in enumerate(names):
                if name in reserved_names:
                    stem, ext = os.path.splitext(name)
                    j = 0
                    new_name = name
                    while new_name in reserved_names:
                        suffix = f"_{now}" if j == 0 else f"_{now}_{j}"
                        new_name = f"{stem}{suffix}{ext}"
                        j += 1
                    names[i] = new_name
                    if conflict_mode == ConflictMode.REPLACE:
                        to_rename[new_name] = name
                    reserved_names.add(new_name)
        upload_tasks.append(
            local_executor.submit(_copy_videos, names=names, ids=ids, metas=metas, infos=batch)
        )

    replace_tasks = []
    rename_executor = ThreadPoolExecutor(max_workers=5)
    src_id_to_dst: Dict[int, object] = {}
    for task in as_completed(upload_tasks):
        src_batch, dst_batch = task.result()
        for s, d in zip(src_batch, dst_batch):
            src_id_to_dst[s.id] = d

        def _maybe_copy_anns_and_replace_videos(sb, db):
            if clone_annotations:
                _copy_anns(sb, db)
            # Video REPLACE: simple swap via rename then remove
            if to_rename:
                for d in db:
                    if d.name in to_rename:
                        orig = to_rename[d.name]
                        if orig in existing:
                            api.video.edit(existing[orig].id, name=orig + "__to_remove")
                            api.video.remove(existing[orig].id)
                            api.video.edit(d.id, name=orig)
            return sb, db

        replace_tasks.append(
            rename_executor.submit(_maybe_copy_anns_and_replace_videos, src_batch, dst_batch)
        )

    for task in as_completed(replace_tasks):
        src_batch, _ = task.result()
        if progress_cb is not None:
            progress_cb(len(src_batch))

    local_executor.shutdown(wait=True)
    rename_executor.shutdown(wait=True)
    return [src_id_to_dst[v.id] for v in src_infos]


def _copy_volume_items(
    api: sly.Api,
    src_infos: List,
    dst_dataset_id: int,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    clone_annotations: bool = True,
    progress_cb=None,
    **kwargs,
) -> List:
    """Clone volumes and their annotations from ``src_infos`` into ``dst_dataset_id``.

    Handles Mask3D geometry uploads and silently skips unsupported
    ``ClosedSurfaceMesh`` geometries with a warning.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_infos: Source ``VolumeInfo`` list.
    :type src_infos: list
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)``.
    :type progress_cb: callable, optional
    :returns: List of destination ``VolumeInfo`` objects in the same order as
        ``src_infos``.
    :rtype: list
    """
    existing = {info.name: info for info in api.volume.get_list(dst_dataset_id)}

    if conflict_mode == ConflictMode.SKIP:
        len_before = len(src_infos)
        src_infos = [v for v in src_infos if v.name not in existing]
        if progress_cb is not None:
            progress_cb(len_before - len(src_infos))

    if not src_infos:
        return []

    src_dataset_id = src_infos[0].dataset_id
    local_executor = ThreadPoolExecutor(max_workers=5)

    def _copy_volumes(names, hashes, metas, infos):
        uploaded = api.volume.upload_hashes(
            dataset_id=dst_dataset_id, names=names, hashes=hashes, metas=metas
        )
        return infos, uploaded

    def _copy_anns(src, dst):
        ann_jsons = api.volume.annotation.download_bulk(src_dataset_id, [info.id for info in src])
        tasks = []
        mask3d_tmp_dir = tempfile.mkdtemp()
        mask_ids = []
        mask_paths = []
        key_id_map = sly.KeyIdMap()
        set_csm_warning = False
        try:
            for ann_json, dst_info in zip(ann_jsons, dst):
                ann = sly.VolumeAnnotation.from_json(ann_json, project_meta, key_id_map)
                sf_idx_to_remove = []
                for idx, sf in enumerate(ann.spatial_figures):
                    figure_id = key_id_map.get_figure_id(sf.key())
                    if sf.geometry.name() == sly.Mask3D.name():
                        mask_ids.append(figure_id)
                        mask_paths.append(os.path.join(mask3d_tmp_dir, sf.key().hex))
                    if sf.geometry.name() == ClosedSurfaceMesh.name():
                        sf_idx_to_remove.append(idx)
                        set_csm_warning = True
                sf_idx_to_remove.reverse()
                for idx in sf_idx_to_remove:
                    ann.spatial_figures.pop(idx)
                tasks.append(
                    local_executor.submit(
                        api.volume.annotation.append,
                        dst_info.id,
                        ann,
                        key_id_map,
                        volume_info=dst_info,
                    )
                )
            # Download all Mask3D geometries after the loop (not inside it)
            if mask_ids:
                api.volume.figure.download_sf_geometries(mask_ids, mask_paths)
            for t in as_completed(tasks):
                t.result()
            if mask_paths:
                progress_masks = tqdm(total=len(mask_paths), desc="Uploading Mask 3D geometries")
                for file in mask_paths:
                    with open(file, "rb") as f:
                        key = UUID(os.path.basename(f.name))
                        api.volume.figure.upload_sf_geometries([key], {key: f.read()}, key_id_map)
                    progress_masks.update(1)
                progress_masks.close()
        finally:
            shutil.rmtree(mask3d_tmp_dir, ignore_errors=True)
        if set_csm_warning:
            logger.warning("Closed Surface Meshes are no longer supported. Skipped copying.")
        return src, dst

    to_rename: Dict[str, str] = {}
    reserved_names = set(existing.keys())
    upload_tasks = []
    for batch in sly.batched(src_infos):
        names = [v.name for v in batch]
        hashes = [v.hash for v in batch]
        metas = [v.meta for v in batch]
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if conflict_mode in (ConflictMode.RENAME, ConflictMode.REPLACE):
            for i, name in enumerate(names):
                if name in reserved_names:
                    stem, ext = os.path.splitext(name)
                    j = 0
                    new_name = name
                    while new_name in reserved_names:
                        suffix = f"_{now}" if j == 0 else f"_{now}_{j}"
                        new_name = f"{stem}{suffix}{ext}"
                        j += 1
                    names[i] = new_name
                    if conflict_mode == ConflictMode.REPLACE:
                        to_rename[new_name] = name
                    reserved_names.add(new_name)
        upload_tasks.append(
            local_executor.submit(
                _copy_volumes, names=names, hashes=hashes, metas=metas, infos=batch
            )
        )

    replace_tasks = []
    rename_executor = ThreadPoolExecutor(max_workers=5)
    src_id_to_dst: Dict[int, object] = {}
    for task in as_completed(upload_tasks):
        src_batch, dst_batch = task.result()
        for s, d in zip(src_batch, dst_batch):
            src_id_to_dst[s.id] = d

        def _maybe_copy_anns_and_replace_volumes(sb, db):
            if clone_annotations:
                _copy_anns(sb, db)
            if to_rename:
                for d in db:
                    if d.name in to_rename:
                        orig = to_rename[d.name]
                        if orig in existing:
                            api.volume.edit(existing[orig].id, name=orig + "__to_remove")
                            api.volume.remove(existing[orig].id)
                            api.volume.edit(d.id, name=orig)
            return sb, db

        replace_tasks.append(
            rename_executor.submit(_maybe_copy_anns_and_replace_volumes, src_batch, dst_batch)
        )

    for task in as_completed(replace_tasks):
        src_batch, _ = task.result()
        if progress_cb is not None:
            progress_cb(len(src_batch))

    local_executor.shutdown(wait=True)
    rename_executor.shutdown(wait=True)
    return [src_id_to_dst[v.id] for v in src_infos]


def _copy_pointcloud_items(
    api: sly.Api,
    src_infos: List,
    dst_dataset_id: int,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    clone_annotations: bool = True,
    progress_cb=None,
    **kwargs,
) -> List:
    """Copy point clouds and their annotations from ``src_infos`` into ``dst_dataset_id``.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_infos: Source ``PointcloudInfo`` list.
    :type src_infos: list
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)``.
    :type progress_cb: callable, optional
    :returns: List of destination ``PointcloudInfo`` objects in the same order as
        ``src_infos``.
    :rtype: list
    """

    existing = {info.name: info for info in api.pointcloud.get_list(dst_dataset_id)}

    if conflict_mode == ConflictMode.SKIP:
        src_infos = [v for v in src_infos if v.name not in existing]

    if not src_infos:
        return []

    src_dataset_id = src_infos[0].dataset_id
    local_executor = ThreadPoolExecutor(max_workers=5)

    def _copy_pcds(names, hashes, metas, infos):
        uploaded = api.pointcloud.upload_hashes(
            dataset_id=dst_dataset_id, names=names, hashes=hashes, metas=metas
        )
        return infos, uploaded

    def _copy_anns(src, dst):
        src_ids = [info.id for info in src]
        dst_ids = [info.id for info in dst]
        ann_jsons = api.pointcloud.annotation.download_bulk(src_dataset_id, src_ids)
        tasks = []
        for ann_json, dst_id in zip(ann_jsons, dst_ids):
            key_id_map = sly.KeyIdMap()
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_meta, key_id_map)
            tasks.append(
                local_executor.submit(api.pointcloud.annotation.append, dst_id, ann, key_id_map)
            )
        for t in as_completed(tasks):
            t.result()
        return src, dst

    to_rename: Dict[str, str] = {}
    reserved_names = set(existing.keys())
    upload_tasks = []
    for batch in sly.batched(src_infos):
        names = [v.name for v in batch]
        hashes = [v.hash for v in batch]
        metas = [v.meta for v in batch]
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if conflict_mode in (ConflictMode.RENAME, ConflictMode.REPLACE):
            for i, name in enumerate(names):
                if name in reserved_names:
                    stem, ext = os.path.splitext(name)
                    j = 0
                    new_name = name
                    while new_name in reserved_names:
                        suffix = f"_{now}" if j == 0 else f"_{now}_{j}"
                        new_name = f"{stem}{suffix}{ext}"
                        j += 1
                    names[i] = new_name
                    if conflict_mode == ConflictMode.REPLACE:
                        to_rename[new_name] = name
                    reserved_names.add(new_name)
        upload_tasks.append(
            local_executor.submit(_copy_pcds, names=names, hashes=hashes, metas=metas, infos=batch)
        )

    replace_tasks = []
    rename_executor = ThreadPoolExecutor(max_workers=5)
    src_id_to_dst: Dict[int, object] = {}
    for task in as_completed(upload_tasks):
        src_batch, dst_batch = task.result()
        for s, d in zip(src_batch, dst_batch):
            src_id_to_dst[s.id] = d

        def _maybe_copy_anns_and_replace_pcd(sb, db):
            if clone_annotations:
                _copy_anns(sb, db)
            if to_rename:
                for d in db:
                    if d.name in to_rename:
                        orig = to_rename[d.name]
                        if orig in existing:
                            api.pointcloud.edit(existing[orig].id, name=orig + "__to_remove")
                            api.pointcloud.remove(existing[orig].id)
                            api.pointcloud.edit(d.id, name=orig)
            return sb, db

        replace_tasks.append(
            rename_executor.submit(_maybe_copy_anns_and_replace_pcd, src_batch, dst_batch)
        )

    for task in as_completed(replace_tasks):
        src_batch, _ = task.result()
        if progress_cb is not None:
            progress_cb(len(src_batch))

    local_executor.shutdown(wait=True)
    rename_executor.shutdown(wait=True)
    return [src_id_to_dst[v.id] for v in src_infos]


def _copy_pointcloud_episode_items(
    api: sly.Api,
    src_infos: List,
    dst_dataset_id: int,
    project_meta: sly.ProjectMeta,
    conflict_mode: ConflictMode = ConflictMode.RENAME,
    clone_annotations: bool = True,
    progress_cb=None,
    **kwargs,
) -> List:
    """Copy point cloud episode frames and annotations into ``dst_dataset_id``.

    Uploads frames (with conflict resolution), copies related images, and
    then appends the full episode annotation using the frame-to-pointcloud
    ID mapping built during the upload phase.

    :param api: Supervisely API instance.
    :type api: sly.Api
    :param src_infos: Source ``PointcloudEpisodeInfo`` list.
    :type src_infos: list
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param project_meta: ``ProjectMeta`` of the source project.
    :type project_meta: sly.ProjectMeta
    :param conflict_mode: Conflict resolution mode. One of ``ConflictMode.SKIP``, ``ConflictMode.RENAME``, ``ConflictMode.REPLACE``.
    :type conflict_mode: ConflictMode
    :param clone_annotations: Whether to copy annotations.
    :type clone_annotations: bool
    :param progress_cb: Optional callable ``(n_items: int)`` called once per
        uploaded frame.
    :type progress_cb: callable, optional
    :returns: List of destination pointcloud episode info objects in the same order
        as ``src_infos``.
    :rtype: list
    """

    existing = {info.name: info for info in api.pointcloud_episode.get_list(dst_dataset_id)}

    if conflict_mode == ConflictMode.SKIP:
        src_infos = [v for v in src_infos if v.name not in existing]

    if not src_infos:
        return []

    src_dataset_id = src_infos[0].dataset_id
    frame_to_pointcloud_ids: Dict[int, int] = {}
    local_executor = ThreadPoolExecutor(max_workers=5)

    def _upload_hashes(infos):
        names = [v.name for v in infos]
        hashes = [v.hash for v in infos]
        metas = [v.meta for v in infos]
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        to_remove_names = []
        to_rename: Dict[str, str] = {}
        # track reserved names within this batch to avoid intra-batch collisions
        reserved = set(existing.keys())
        for i, name in enumerate(names):
            if name in reserved:
                stem, ext = os.path.splitext(name)
                j = 0
                new_name = name
                while new_name in reserved:
                    suffix = f"_{now}" if j == 0 else f"_{now}_{j}"
                    new_name = f"{stem}{suffix}{ext}"
                    j += 1
                names[i] = new_name
                reserved.add(new_name)
                if conflict_mode == ConflictMode.REPLACE:
                    to_remove_names.append(name)
                    to_rename[new_name] = name
        dst_batch = api.pointcloud_episode.upload_hashes(
            dataset_id=dst_dataset_id, names=names, hashes=hashes, metas=metas
        )
        if to_remove_names:
            rm_ids = [existing[n].id for n in to_remove_names if n in existing]
            if rm_ids:
                api.pointcloud_episode.remove_batch(rm_ids)
        if to_rename:
            rename_tasks = []
            for dst_info in dst_batch:
                if dst_info.name in to_rename:
                    rename_tasks.append(
                        local_executor.submit(
                            api.pointcloud_episode.edit,
                            dst_info.id,
                            name=to_rename[dst_info.name],
                        )
                    )
            updated = []
            for t in as_completed(rename_tasks):
                updated.append(t.result())
            for upd in updated:
                dst_batch = [upd if upd.id == d.id else d for d in dst_batch]
        return {src.id: dst for src, dst in zip(infos, dst_batch)}

    def _upload_single(src_id, dst_info):
        frame_to_pointcloud_ids[dst_info.meta["frame"]] = dst_info.id
        rel_images = api.pointcloud_episode.get_list_related_images(id=src_id)
        if rel_images:
            rimg_infos = [
                {
                    sly.api.ApiField.ENTITY_ID: dst_info.id,
                    sly.api.ApiField.NAME: rel[sly.api.ApiField.NAME],
                    sly.api.ApiField.HASH: rel[sly.api.ApiField.HASH],
                    sly.api.ApiField.META: rel[sly.api.ApiField.META],
                }
                for rel in rel_images
            ]
            api.pointcloud_episode.add_related_images(rimg_infos)
        if progress_cb is not None:
            progress_cb(1)

    upload_tasks = [
        local_executor.submit(_upload_hashes, batch) for batch in sly.batched(src_infos)
    ]

    dst_infos_dict: Dict[int, object] = {}
    if not clone_annotations:
        for task in as_completed(upload_tasks):
            batch_result = task.result()
            dst_infos_dict.update(batch_result)
            if progress_cb is not None:
                progress_cb(len(batch_result))
        local_executor.shutdown(wait=True)
        return [dst_infos_dict[v.id] for v in src_infos]

    key_id_map = sly.KeyIdMap()
    ann_json = api.pointcloud_episode.annotation.download(src_dataset_id)
    ann = sly.PointcloudEpisodeAnnotation.from_json(
        data=ann_json, project_meta=project_meta, key_id_map=key_id_map
    )

    ann_tasks = []
    for task in as_completed(upload_tasks):
        src_to_dst = task.result()
        dst_infos_dict.update(src_to_dst)
        for src_id, dst_info in src_to_dst.items():
            ann_tasks.append(local_executor.submit(_upload_single, src_id, dst_info))

    for t in as_completed(ann_tasks):
        t.result()

    api.pointcloud_episode.annotation.append(
        dataset_id=dst_dataset_id,
        ann=ann,
        frame_to_pointcloud_ids=frame_to_pointcloud_ids,
        key_id_map=key_id_map,
    )

    local_executor.shutdown(wait=True)
    return [dst_infos_dict[v.id] for v in src_infos]
