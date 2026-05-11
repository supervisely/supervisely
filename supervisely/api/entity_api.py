# coding: utf-8
"""Generic entity API for Supervisely — works across all entity types (images, videos, point clouds, etc.)."""

from __future__ import annotations

import asyncio

import aiofiles
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Union,
)

from tqdm import tqdm

from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.io.fs import ensure_base_path, get_file_hash_async


_CREATED_BY_FIELD = ApiField.CREATED_BY_ID[0][0]


class EntityInfo(NamedTuple):
    """
    NamedTuple with entity information from Supervisely.

    :Usage Example:

        .. code-block:: python

            import supervisely as sly
            api = sly.Api.from_env()
            entity = api.entity.get_info_by_id(entity_id)
            print(entity)
            # EntityInfo(id=12345, name="image.jpg", dataset_id=678, ...)
    """

    #: int: Entity ID in Supervisely.
    id: int

    #: str: Entity name (filename). Populated by ``entities.bulk.add`` and similar endpoints.
    name: str

    #: str: Entity description.
    description: Optional[str] = None

    #: int: Parent entity ID.
    #:
    #: Used to associate this entity with a primary entity:
    #:
    #: - **Point cloud photo context**: set to the point cloud entity ID to attach
    #:   context images (e.g. camera frames) to a point cloud.
    #: - **Image overlay**: set to the base image entity ID to attach overlay images
    #:   that are rendered on top of it.
    parent_id: Optional[int] = None

    #: int: :class:`~supervisely.project.project.Project` ID in Supervisely.
    project_id: Optional[int] = None

    #: int: :class:`~supervisely.project.project.Dataset` ID in Supervisely.
    dataset_id: Optional[int] = None

    #: str: Entity creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: Optional[str] = None

    #: str: Time of last entity update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: Optional[str] = None

    #: int: ID of the user who created the entity.
    created_by: Optional[int] = None

    #: dict: Custom additional entity metadata.
    meta: Optional[dict] = None

    #: dict: File metadata (mime type, size, width, height, etc.).
    file_meta: Optional[dict] = None

    #: int: Frame number (for video frames or point cloud episodes).
    frame: Optional[int] = None

    #: str: Entity hash obtained by base64(sha256(file_content)).
    hash: Optional[str] = None

    #: str: Relative storage URL to entity file.
    #: e.g. "/h5un6l2bnaz1vj8a9qgms4-public/images/original/7/h/Vo/...jpg".
    path_original: Optional[str] = None

    #: str: Use link as source for entities stored at a remote server.
    #: e.g. "http://your-server/image1.jpg".
    link: Optional[str] = None

    #: str: Full storage URL to entity file.
    #: e.g. "http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg".
    full_storage_url: Optional[str] = None

    #: int: Number of annotation objects in the entity.
    objects_count: Optional[int] = None

    #: int: :class:`~supervisely.api.workspace_api.WorkspaceApi` ID in Supervisely.
    workspace_id: Optional[int] = None

    #: int: Entity file size in bytes.
    size: Optional[int] = None

    #: dict: Custom data associated with the entity.
    custom_data: Optional[dict] = None

    #: any: Value used for custom sorting of entities.
    custom_sort: Optional[Any] = None

    #: int: Position of the entity within an entities collection.
    collection_item_index: Optional[int] = None

    # DO NOT DELETE THIS COMMENT
    #! New fields must be added with default values to keep backward compatibility.


class EntityDescriptor:
    """
    Descriptor for a single entity to be added to a dataset via :meth:`EntityApi.add`.

    Use the class methods to construct a descriptor from the appropriate source type,
    then pass a list of descriptors (or plain dicts) to :meth:`EntityApi.add`.

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            entities = [
                sly.EntityDescriptor.from_hash("ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=", name="img.jpg"),
                sly.EntityDescriptor.from_link("http://my-server/image.jpg", name="remote.jpg"),
                sly.EntityDescriptor.from_entity_id(12345, name="copy.jpg"),
            ]
            api.entity.add(dataset_id=678, entities=entities)
    """

    def __init__(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        meta: Optional[dict] = None,
        parent_id: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_by: Optional[int] = None,
        **_source: Any,
    ):
        self._data: Dict[str, Any] = {ApiField.NAME: name}
        self._data.update(_source)
        if description is not None:
            self._data[ApiField.DESCRIPTION] = description
        if meta is not None:
            self._data[ApiField.META] = meta
        if parent_id is not None:
            self._data[ApiField.PARENT_ID] = parent_id
        if created_at is not None:
            self._data[ApiField.CREATED_AT] = created_at
        if updated_at is not None:
            self._data[ApiField.UPDATED_AT] = updated_at
        if created_by is not None:
            self._data[_CREATED_BY_FIELD] = created_by

    @classmethod
    def from_hash(cls, hash: str, name: str, **kwargs) -> "EntityDescriptor":
        """
        Create a descriptor for an entity already stored in Supervisely by its hash.

        :param hash: Base64-encoded SHA-256 hash of the file content.
        :type hash: str
        :param name: Display name for the entity.
        :type name: str
        :param kwargs: Optional fields: ``description``, ``meta``, ``created_at``,
            ``updated_at``, ``created_by``, and ``parent_id`` — ID of the primary
            entity this one is attached to (point cloud ID for photo context images,
            base image ID for overlays).
        :returns: :class:`EntityDescriptor` instance.
        :rtype: :class:`EntityDescriptor`
        """
        return cls(name=name, **{ApiField.HASH: hash}, **kwargs)

    @classmethod
    def from_link(cls, link: str, name: str, **kwargs) -> "EntityDescriptor":
        """
        Create a descriptor for an entity stored on a remote server.

        :param link: URL of the remote file.
        :type link: str
        :param name: Display name for the entity.
        :type name: str
        :param kwargs: Optional fields: ``description``, ``meta``, ``created_at``,
            ``updated_at``, ``created_by``, and ``parent_id`` — ID of the primary
            entity this one is attached to (point cloud ID for photo context images,
            base image ID for overlays).
        :returns: :class:`EntityDescriptor` instance.
        :rtype: :class:`EntityDescriptor`
        """
        return cls(name=name, **{ApiField.LINK: link}, **kwargs)

    @classmethod
    def from_entity_id(cls, entity_id: int, name: str, **kwargs) -> "EntityDescriptor":
        """
        Create a descriptor that copies an existing entity into a new dataset.

        :param entity_id: ID of the source entity.
        :type entity_id: int
        :param name: Display name for the new entity.
        :type name: str
        :param kwargs: Optional fields: ``description``, ``meta``, ``created_at``,
            ``updated_at``, ``created_by``, and ``parent_id`` — ID of the primary
            entity this one is attached to (point cloud ID for photo context images,
            base image ID for overlays).
        :returns: :class:`EntityDescriptor` instance.
        :rtype: :class:`EntityDescriptor`
        """
        return cls(name=name, **{ApiField.ENTITY_ID: entity_id}, **kwargs)

    @classmethod
    def from_team_file(cls, team_file_id: int, name: str, **kwargs) -> "EntityDescriptor":
        """
        Create a descriptor for a file stored in Supervisely team storage.

        :param team_file_id: ID of the file in team storage.
        :type team_file_id: int
        :param name: Display name for the entity.
        :type name: str
        :param kwargs: Optional fields: ``description``, ``meta``, ``created_at``,
            ``updated_at``, ``created_by``, and ``parent_id`` — ID of the primary
            entity this one is attached to (point cloud ID for photo context images,
            base image ID for overlays).
        :returns: :class:`EntityDescriptor` instance.
        :rtype: :class:`EntityDescriptor`
        """
        return cls(name=name, **{ApiField.TEAM_FILE_ID: team_file_id}, **kwargs)

    @classmethod
    def from_source_blob(
        cls,
        blob_id: int,
        offset_start: int,
        offset_end: int,
        name: str,
        **kwargs,
    ) -> "EntityDescriptor":
        """
        Create a descriptor for a blob-backed entity defined by a byte range.

        :param blob_id: ID of the source local entity.
        :type blob_id: int
        :param offset_start: Start byte offset within the blob.
        :type offset_start: int
        :param offset_end: End byte offset within the blob.
        :type offset_end: int
        :param name: Display name for the entity.
        :type name: str
        :param kwargs: Optional fields: ``description``, ``meta``, ``created_at``,
            ``updated_at``, ``created_by``, and ``parent_id`` — ID of the primary
            entity this one is attached to (point cloud ID for photo context images,
            base image ID for overlays).
        :returns: :class:`EntityDescriptor` instance.
        :rtype: :class:`EntityDescriptor`
        """
        source_blob = {
            ApiField.OFFSET_START: offset_start,
            ApiField.OFFSET_END: offset_end,
        }
        return cls(
            name=name,
            **{ApiField.ENTITY_ID: blob_id, ApiField.SOURCE_BLOB: source_blob},
            **kwargs,
        )

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the descriptor as a plain dict suitable for the API request."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"EntityDescriptor({self._data!r})"


class EntityApi(ModuleApiBase):
    """
    API for working with generic entities in Supervisely.

    Provides a unified interface for entities across all project types
    (images, videos, point clouds, volumes, etc.) using a generic
    id-based contract.

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            # List entities in a dataset
            entities = api.entity.get_list(dataset_id=123)
            for entity in entities:
                print(entity.id, entity.name)

            # Get a single entity's info
            entity = api.entity.get_info_by_id(id=456)

            # Download entity file content to disk
            api.entity.download(id=456, path="output.bin")

            # Download entity file content to memory
            data = api.entity.download_bytes(id=456)
    """

    @staticmethod
    def info_sequence() -> List[str]:
        """
        Sequence of API field names that represent :class:`EntityInfo`.

        :returns: List of API field name strings.
        :rtype: List[str]
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.PARENT_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            _CREATED_BY_FIELD,
            ApiField.META,
            ApiField.FILE_META,
            ApiField.FRAME,
            ApiField.HASH,
            ApiField.PATH_ORIGINAL,
            ApiField.LINK,
            ApiField.FULL_STORAGE_URL,
            ApiField.OBJECTS_COUNT,
            ApiField.WORKSPACE_ID,
            ApiField.SIZE,
            ApiField.CUSTOM_DATA,
            ApiField.CUSTOM_SORT,
            ApiField.COLLECTION_ITEM_INDEX,
        ]

    @staticmethod
    def info_tuple_name() -> str:
        """
        Name of the NamedTuple that represents entity info.

        :returns: NamedTuple class name.
        :rtype: str
        """
        return "EntityInfo"

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True) -> Optional[EntityInfo]:
        """Convert server JSON into EntityInfo.

        Extends the base implementation with two normalizations:
        - ``frame``: falls back to ``meta["frame"]`` when absent at the top level
          (point cloud episode entities store the frame index inside ``meta``).
        - ``size``: coerced to ``int`` from ``fileMeta["size"]`` when absent at the
          top level (some older endpoints return size only inside the file-meta blob).
        """
        if info is None:
            return None

        # Normalize before delegating to the generic base implementation.
        normalized = info
        meta = info.get(ApiField.META)
        file_meta = info.get(ApiField.FILE_META)

        frame_missing = info.get(ApiField.FRAME) is None
        size_missing = info.get(ApiField.SIZE) is None

        if frame_missing or size_missing:
            normalized = dict(info)
            if frame_missing and isinstance(meta, dict):
                frame = meta.get(ApiField.FRAME)
                if frame is not None:
                    normalized[ApiField.FRAME] = frame
            if size_missing and isinstance(file_meta, dict):
                raw_size = file_meta.get(ApiField.SIZE)
                if raw_size is not None:
                    try:
                        normalized[ApiField.SIZE] = int(raw_size)
                    except (TypeError, ValueError):
                        pass

        return super()._convert_json_info(normalized, skip_missing=skip_missing)

    @staticmethod
    def _normalize_entity_payload(entity: Union[Dict, EntityDescriptor]) -> Dict:
        payload = entity.to_dict() if isinstance(entity, EntityDescriptor) else dict(entity)

        if "created_by" in payload and _CREATED_BY_FIELD not in payload:
            payload[_CREATED_BY_FIELD] = payload.pop("created_by")

        source_blob = payload.get(ApiField.SOURCE_BLOB)
        if isinstance(source_blob, dict):
            if "offset_start" in source_blob and ApiField.OFFSET_START not in source_blob:
                source_blob[ApiField.OFFSET_START] = source_blob.pop("offset_start")
            if "offset_end" in source_blob and ApiField.OFFSET_END not in source_blob:
                source_blob[ApiField.OFFSET_END] = source_blob.pop("offset_end")

        return payload

    def get_info_by_id(
        self,
        id: int,
        fields: Optional[List[str]] = None,
        omit_frames_to_timecodes: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
    ) -> Optional[EntityInfo]:
        """
        Get entity information by its ID.

        Calls ``entities.info``.

        :param id: Entity ID in Supervisely.
        :type id: int
        :param fields: List of fields to include in the response. If None, all default
            fields are returned. Available values: "id", "name", "description",
            "parentId", "projectId", "datasetId", "createdAt", "updatedAt",
            "createdBy", "meta", "fileMeta", "frame", "hash", "pathOriginal",
            "link", "fullStorageUrl", "customData", "objectsCount",
            "workspaceId", "size", "customSort", "collectionItemIndex".
        :type fields: List[str], optional
        :param omit_frames_to_timecodes: If True, omit the frames-to-timecodes mapping
            from the response (useful for large video entities).
        :type omit_frames_to_timecodes: bool, optional
        :param force_metadata_for_links: If True, updates metadata for entities with
            remote storage links.
        :type force_metadata_for_links: bool, optional
        :returns: :class:`EntityInfo` if the entity is found, otherwise None.
        :rtype: :class:`EntityInfo` or None

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()
                entity = api.entity.get_info_by_id(id=12345)
                if entity is not None:
                    print(entity.name, entity.dataset_id)
        """
        data = {ApiField.ID: id}
        if fields is not None:
            data[ApiField.FIELDS] = fields
        if omit_frames_to_timecodes:
            data[ApiField.OMIT_FRAMES_TO_TIMECODES] = omit_frames_to_timecodes
        data[ApiField.FORCE_METADATA_FOR_LINKS] = force_metadata_for_links
        response = self._api.post("entities.info", data)
        return self._convert_json_info(response.json())

    def get_list(
        self,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
        filters: Optional[List[Dict]] = None,
        sort: Literal[
            "id",
            "name",
            "description",
            "objectsCount",
            "datasetId",
            "createdAt",
            "updatedAt",
            "size",
            "customSort",
            "collectionItemIndex",
        ] = "id",
        sort_order: Literal["asc", "desc"] = "asc",
        limit: Optional[int] = None,
        fields: Optional[List[str]] = None,
        extra_fields: Optional[List[str]] = None,
        recursive: Optional[bool] = False,
        show_disabled: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[EntityInfo]:
        """
        Get a list of entities from a dataset or project.

        Calls ``entities.list``.

        :param dataset_id: Dataset ID to list entities from. Mutually exclusive
            with *project_id*.
        :type dataset_id: int, optional
        :param project_id: Project ID to list entities from. Mutually exclusive
            with *dataset_id*.
        :type project_id: int, optional
        :param filters: Filters to apply to the list. Each filter is a dict with
            keys "field", "operator", and "value".
        :type filters: List[Dict], optional
        :param sort: Field to sort entities by. One of {"id" (default), "name",
            "description", "objectsCount", "datasetId", "createdAt", "updatedAt",
            "size", "customSort", "collectionItemIndex"}.
        :type sort: str, optional
        :param sort_order: Sort direction. One of {"asc" (default), "desc"}.
        :type sort_order: str, optional
        :param limit: Maximum number of entities to return. Returns all if None.
        :type limit: int, optional
        :param fields: List of fields to include in each entity's response.
        :type fields: List[str], optional
        :param extra_fields: Additional fields to include beyond the default set.
        :type extra_fields: List[str], optional
        :param recursive: If True, recursively list entities in nested datasets.
        :type recursive: bool, optional
        :param show_disabled: If True, include disabled entities.
        :type show_disabled: bool, optional
        :param progress_cb: Function or tqdm instance to track progress.
        :type progress_cb: tqdm or callable, optional
        :returns: List of :class:`EntityInfo` objects.
        :rtype: List[:class:`EntityInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                from tqdm import tqdm

                api = sly.Api.from_env()

                entities = api.entity.get_list(dataset_id=678)
                print(len(entities))

                # With progress tracking
                with tqdm(total=None, desc="Fetching entities") as pbar:
                    entities = api.entity.get_list(
                        project_id=42,
                        sort="createdAt",
                        sort_order="desc",
                        progress_cb=pbar.update,
                    )
        """
        if dataset_id is None and project_id is None:
            raise ValueError("Either 'dataset_id' or 'project_id' must be specified.")

        data = {}
        if dataset_id is not None:
            data[ApiField.DATASET_ID] = dataset_id
        if project_id is not None:
            data[ApiField.PROJECT_ID] = project_id
        if filters is not None:
            data[ApiField.FILTERS] = filters
        data[ApiField.SORT] = sort
        data[ApiField.SORT_ORDER] = sort_order
        if fields is not None:
            data[ApiField.FIELDS] = fields
        if extra_fields is not None:
            data[ApiField.EXTRA_FIELDS] = extra_fields
        if recursive:
            data[ApiField.RECURSIVE] = recursive
        if show_disabled:
            data[ApiField.SHOW_DISABLED] = show_disabled

        return self.get_list_all_pages(
            "entities.list",
            data,
            progress_cb=progress_cb,
            limit=limit,
        )

    def get_list_generator(
        self,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
        filters: Optional[List[Dict]] = None,
        sort: Literal[
            "id",
            "name",
            "description",
            "objectsCount",
            "datasetId",
            "createdAt",
            "updatedAt",
            "size",
            "customSort",
            "collectionItemIndex",
        ] = "id",
        sort_order: Literal["asc", "desc"] = "asc",
        limit: Optional[int] = None,
        fields: Optional[List[str]] = None,
        extra_fields: Optional[List[str]] = None,
        recursive: Optional[bool] = False,
        batch_size: Optional[int] = 500,
        show_disabled: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Iterator[List[EntityInfo]]:
        """
        Return a generator that yields batches of entities from a dataset or project.

        Uses token-based pagination for efficient large-dataset traversal.

        Calls ``entities.list``.

        :param dataset_id: Dataset ID to list entities from. Mutually exclusive
            with *project_id*.
        :type dataset_id: int, optional
        :param project_id: Project ID to list entities from. Mutually exclusive
            with *dataset_id*.
        :type project_id: int, optional
        :param filters: Filters to apply to the list.
        :type filters: List[Dict], optional
        :param sort: Field to sort entities by (e.g. "id", "name", "createdAt"). Sorting is not guaranteed when
            using token-based pagination.
        :type sort: str, optional
        :param sort_order: Sort direction. One of {"asc" (default), "desc"}.
        :type sort_order: str, optional
        :param limit: Maximum total number of entities to yield. Yields all if None.
        :type limit: int, optional
        :param fields: List of fields to include in each entity's response.
        :type fields: List[str], optional
        :param extra_fields: Additional fields to include beyond the default set.
        :type extra_fields: List[str], optional
        :param recursive: If True, recursively list entities in nested datasets.
        :type recursive: bool, optional
        :param batch_size: Number of entities per batch request.
        :type batch_size: int, optional
        :param progress_cb: Function or tqdm instance to track progress.
        :type progress_cb: tqdm or callable, optional
        :returns: Generator yielding lists of :class:`EntityInfo` objects.
        :rtype: Iterator[List[:class:`EntityInfo`]]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                for batch in api.entity.get_list_generator(dataset_id=678, batch_size=100):
                    for entity in batch:
                        print(entity.id, entity.name)
        """
        if dataset_id is None and project_id is None:
            raise ValueError("Either 'dataset_id' or 'project_id' must be specified.")

        data = {
            ApiField.PAGINATION_MODE: ApiField.TOKEN,
            ApiField.PER_PAGE: batch_size,
        }
        if dataset_id is not None:
            data[ApiField.DATASET_ID] = dataset_id
        if project_id is not None:
            data[ApiField.PROJECT_ID] = project_id
        if filters is not None:
            data[ApiField.FILTERS] = filters
        data[ApiField.SORT] = sort
        data[ApiField.SORT_ORDER] = sort_order
        if fields is not None:
            data[ApiField.FIELDS] = fields
        if extra_fields is not None:
            data[ApiField.EXTRA_FIELDS] = extra_fields
        if recursive:
            data[ApiField.RECURSIVE] = recursive
        if show_disabled:
            data[ApiField.SHOW_DISABLED] = show_disabled

        yield from self.get_list_all_pages_generator(
            "entities.list",
            data,
            progress_cb=progress_cb,
            limit=limit,
        )

    def add(
        self,
        dataset_id: int,
        entities: List[Union[Dict, EntityDescriptor]],
        generate_unique_names: Optional[bool] = False,
        force_metadata_for_links: Optional[bool] = True,
        skip_validation: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[EntityInfo]:
        """
        Add entities to a dataset using a generic ID-based contract.

        Each entity dict may contain any of the following keys:

        - ``hash`` (str): file hash (base64-encoded SHA-256) for entities already
          stored in Supervisely.
        - ``link`` (str): URL for entities stored on a remote server.
        - ``entityId`` (int): ID of an existing entity to copy into this dataset.
        - ``teamFileId`` (int): ID of a file in Supervisely team storage.
        - ``name`` (str): display name / filename for the entity.
        - ``description`` (str): optional description.
        - ``meta`` (dict): optional metadata.
        - ``parentId`` (int): ID of the primary entity this one is attached to.

          - **Point cloud photo context**: set to the point cloud entity ID to attach
            context images (e.g. camera frames) to a point cloud.
          - **Image overlay**: set to the base image entity ID to attach overlay images
            that are rendered on top of it.

          Omit for standalone entities.
        - ``createdAt`` (str): optional creation timestamp (ISO 8601).
        - ``updatedAt`` (str): optional update timestamp (ISO 8601).
        - ``createdBy`` (int): optional ID of the creating user.
        - ``sourceBlob`` (dict): for blob-backed entities with ``offsetStart`` /
          ``offsetEnd`` byte offsets.

        Calls ``entities.bulk.add``.

        :param dataset_id: ID of the destination dataset.
        :type dataset_id: int
        :param entities: List of entity descriptor dicts (see above).
        :type entities: List[Dict]
        :param generate_unique_names: If True, automatically rename entities when
            a name conflict occurs in the target dataset.
        :type generate_unique_names: bool, optional
        :param force_metadata_for_links: If True, fetch and store metadata for
            entities added via remote link.
        :type force_metadata_for_links: bool, optional
        :param skip_validation: If True, skip server-side validation of the entity
            descriptors.
        :type skip_validation: bool, optional
        :param progress_cb: Function or tqdm instance to track progress.
        :type progress_cb: tqdm or callable, optional
        :returns: List of :class:`EntityInfo` objects for the newly added entities.
        :rtype: List[:class:`EntityInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                # Add entity by hash
                new_entities = api.entity.add(
                    dataset_id=678,
                    entities=[
                        {"hash": "ZdpMD+ZMJx0R8BgsCzJcqM7qP4M8f1AEtoYc87xZmyQ=", "name": "img.jpg"},
                    ],
                )
                print(new_entities[0].id, new_entities[0].name)

                # Copy existing entity from another dataset
                new_entities = api.entity.add(
                    dataset_id=678,
                    entities=[{"entityId": 12345, "name": "copy_of_img.jpg"}],
                )
        """
        data = {
            ApiField.DATASET_ID: dataset_id,
            ApiField.ENTITIES: [self._normalize_entity_payload(e) for e in entities],
            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
            ApiField.SKIP_VALIDATION: skip_validation,
        }
        if generate_unique_names:
            data[ApiField.GENERATE_UNIQUE_NAMES] = generate_unique_names
        response = self._api.post("entities.bulk.add", data)
        result = [self._convert_json_info(item) for item in response.json()]

        if progress_cb is not None:
            progress_cb(len(result))

        return result

    def _download(self, id: int, is_stream: Optional[bool] = False):
        """Private method. Download entity with given ID."""
        return self._api.post("entities.download", {ApiField.ID: id}, stream=is_stream)

    async def _download_async(
        self,
        id: int,
        is_stream: bool = False,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncGenerator:
        """
        Download entity with given ID asynchronously.

        :param id: Entity ID in Supervisely.
        :type id: int
        :param is_stream: If True, returns stream of bytes, otherwise returns response object.
        :type is_stream: bool, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :returns: Stream of bytes or response object.
        :rtype: AsyncGenerator
        """
        api_method_name = "entities.download"
        json_body = {ApiField.ID: id}

        if is_stream:
            async for chunk, hhash in self._api.stream_async(
                api_method_name,
                "POST",
                json_body,
                headers=headers,
                range_start=range_start,
                range_end=range_end,
                chunk_size=chunk_size,
            ):
                yield chunk, hhash
        else:
            response = await self._api.post_async(api_method_name, json_body, headers=headers)
            yield response

    def download_path(self, id: int, path: str) -> None:
        """
        Download entity file content by its ID and save it to local path.

        :param id: Entity ID in Supervisely.
        :type id: int
        :param path: Local file path to save entity bytes.
        :type path: str
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                # Download entity to local file
                api.entity.download_path(id=12345, path="output.bin")
        """
        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    def download(self, id: int, path: str) -> None:
        """Alias for :meth:`download_path` to match other media APIs."""
        return self.download_path(id=id, path=path)

    async def download_path_async(
        self,
        id: int,
        path: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Download entity with given ID to local path asynchronously.

        :param id: Entity ID in Supervisely.
        :type id: int
        :param path: Local save path for entity file.
        :type path: str
        :param semaphore: Semaphore for limiting simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param range_start: Start byte of range for partial download.
        :type range_start: int, optional
        :param range_end: End byte of range for partial download.
        :type range_end: int, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded file.
            Check is not supported for partial downloads.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :returns: None
        :rtype: None
        """
        if range_start is not None or range_end is not None:
            check_hash = False
            headers = headers or {}
            headers["Range"] = f"bytes={range_start or ''}-{range_end or ''}"

        writing_method = "ab" if range_start not in [0, None] else "wb"

        ensure_base_path(path)
        hash_to_check = None
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async with semaphore:
            async with aiofiles.open(path, writing_method) as file:
                async for chunk, hhash in self._download_async(
                    id,
                    is_stream=True,
                    range_start=range_start,
                    range_end=range_end,
                    headers=headers,
                    chunk_size=chunk_size,
                ):
                    await file.write(chunk)
                    hash_to_check = hhash
                    if progress_cb is not None and progress_cb_type == "size":
                        progress_cb(len(chunk))

        if check_hash and hash_to_check is not None:
            downloaded_hash = await get_file_hash_async(path)
            if downloaded_hash != hash_to_check:
                raise RuntimeError(
                    f"Downloaded hash of entity with ID:{id} does not match the expected hash: "
                    f"{downloaded_hash} != {hash_to_check}"
                )

        if progress_cb is not None and progress_cb_type == "number":
            progress_cb(1)

    async def download_paths_async(
        self,
        ids: List[int],
        paths: List[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict] = None,
        chunk_size: int = 1024 * 1024,
        check_hash: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        progress_cb_type: Literal["number", "size"] = "number",
    ) -> None:
        """
        Download entities with given IDs and save them to local paths asynchronously.

        :param ids: List of entity IDs in Supervisely.
        :type ids: List[int]
        :param paths: Local save paths for entities.
        :type paths: List[str]
        :param semaphore: Semaphore for limiting simultaneous downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param headers: Headers for request.
        :type headers: dict, optional
        :param chunk_size: Size of chunk for partial download. Default is 1MB.
        :type chunk_size: int, optional
        :param check_hash: If True, checks hash of downloaded files.
        :type check_hash: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param progress_cb_type: Type of progress callback. Can be "number" or "size".
        :type progress_cb_type: Literal["number", "size"], optional
        :raises ValueError: if len(ids) != len(paths)
        :returns: None
        :rtype: None
        """
        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise ValueError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        tasks = []
        for entity_id, entity_path in zip(ids, paths):
            task = self.download_path_async(
                entity_id,
                entity_path,
                semaphore=semaphore,
                headers=headers,
                chunk_size=chunk_size,
                check_hash=check_hash,
                progress_cb=progress_cb,
                progress_cb_type=progress_cb_type,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    def download_bytes(self, id: int) -> bytes:
        """
        Download entity file content by its ID to memory.

        :param id: Entity ID in Supervisely.
        :type id: int
        :returns: Raw bytes of the entity file.
        :rtype: bytes

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                api = sly.Api.from_env()

                # Download entity to memory
                file_bytes = api.entity.download_bytes(id=12345)
        """
        response = self._download(id)
        return response.content
