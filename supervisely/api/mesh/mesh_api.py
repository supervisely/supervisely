# coding: utf-8
from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Callable, Dict, List, NamedTuple, Optional, Union
from urllib.parse import urlparse

from tqdm import tqdm

from supervisely._utils import batched, generate_free_name, rand_str
from supervisely.api.mesh.mesh_annotation_api import MeshAnnotationAPI
from supervisely.api.mesh.mesh_object_api import MeshObjectApi
from supervisely.api.mesh.mesh_tag_api import MeshTagApi
from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi, _get_single_item
from supervisely.io.fs import ensure_base_path, get_file_ext


ALLOWED_MESH_EXTENSIONS = {".ply", ".stl", ".obj"}


class MeshInfo(NamedTuple):
    """
    NamedTuple with mesh entity information from Supervisely.

    :Usage Example:

        .. code-block:: python

            MeshInfo(
                id=1,
                name="scan.stl",
                title="scan.stl",
                description="",
                parent_id=None,
                workspace_id=2,
                project_id=3,
                dataset_id=4,
                path_original=None,
                full_storage_url="https://app.supervisely.com/.../scan.stl",
                link=None,
                meta={},
                file_meta={},
                frame=None,
                size=None,
                custom_data={},
                objects_count=0,
                tags=[],
                created_by_id=5,
                created_at="2026-01-01T00:00:00.000Z",
                updated_at="2026-01-01T00:00:00.000Z",
            )
    """

    #: int: Mesh ID in Supervisely.
    id: int

    #: str: Mesh filename.
    name: str

    #: str: Display title of the mesh (mirrors :attr:`name`).
    title: str

    #: str: Mesh description.
    description: str

    #: int: ID of the parent entity, if any.
    parent_id: int

    #: int: :class:`~supervisely.api.workspace_api.WorkspaceApi` ID in Supervisely.
    workspace_id: int

    #: int: :class:`~supervisely.project.project.Project` ID in Supervisely.
    project_id: int

    #: int: :class:`~supervisely.project.project.Dataset` ID in Supervisely.
    dataset_id: int

    #: str: Relative storage path to the mesh file.
    path_original: str

    #: str: Absolute storage URL of the mesh file.
    full_storage_url: str

    #: str: External link to the mesh file, if uploaded via URL.
    link: str

    #: dict: Arbitrary metadata dictionary associated with the mesh.
    meta: dict

    #: dict: Low-level file metadata returned by the storage backend.
    file_meta: dict

    #: int: Frame index within a sequence, if applicable.
    frame: int

    #: int: File size in bytes.
    size: int

    #: dict: User-defined custom data blob attached to the mesh.
    custom_data: dict

    #: int: Number of annotation objects attached to this mesh.
    objects_count: int

    #: list: Mesh :class:`~supervisely.mesh_annotation.mesh_tag.MeshTag` server rows.
    tags: list

    #: int: ID of the user who created the mesh.
    created_by_id: int

    #: str: ISO 8601 creation timestamp. e.g. "2026-01-01T00:00:00.000Z".
    created_at: str

    #: str: ISO 8601 last-update timestamp. e.g. "2026-01-01T00:00:00.000Z".
    updated_at: str


class MeshApi(RemoveableBulkModuleApi):
    """API for working with mesh entities in Supervisely."""

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()
                mesh_info = api.mesh.get_info_by_id(mesh_id)
        """
        super().__init__(api)
        if not hasattr(api, "mesh"):
            api.mesh = self
        self.annotation = MeshAnnotationAPI(api)
        self.object = MeshObjectApi(api)
        self.tag = MeshTagApi(api)

    @staticmethod
    def info_sequence():
        """
        Get list of all :class:`~supervisely.api.mesh.mesh_api.MeshInfo` field names.

        :returns: List of MeshInfo field names.
        :rtype: List[str]
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TITLE,
            ApiField.DESCRIPTION,
            ApiField.PARENT_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.LINK,
            ApiField.META,
            ApiField.FILE_META,
            ApiField.FRAME,
            ApiField.SIZE,
            ApiField.CUSTOM_DATA,
            ApiField.OBJECTS_COUNT,
            ApiField.TAGS,
            ApiField.CREATED_BY_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`~supervisely.api.mesh.mesh_api.MeshInfo` NamedTuple.

        :returns: NamedTuple name.
        :rtype: str
        """
        return "MeshInfo"

    @staticmethod
    def default_fields() -> List[str]:
        """
        Default list of fields requested when listing or fetching mesh entities.

        :returns: List of field names.
        :rtype: List[str]
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TITLE,
            ApiField.DESCRIPTION,
            ApiField.PARENT_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.PATH_ORIGINAL,
            ApiField.FULL_STORAGE_URL,
            ApiField.LINK,
            ApiField.META,
            ApiField.FILE_META,
            ApiField.FRAME,
            ApiField.SIZE,
            ApiField.CUSTOM_DATA,
            ApiField.OBJECTS_COUNT,
            ApiField.TAGS,
            "createdBy",
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        """
        Convert a raw JSON dict from the API into a :class:`MeshInfo` NamedTuple.

        Fills in ``name`` from ``title`` (and vice versa) when one of them is missing.

        :param info: Raw mesh info dict from the API.
        :type info: dict
        :param skip_missing: If ``True``, missing fields are tolerated.
        :type skip_missing: bool, optional
        :returns: Parsed mesh info, or ``None`` if *info* is ``None``.
        :rtype: :class:`~supervisely.api.mesh.mesh_api.MeshInfo`
        """
        if info is None:
            return None
        info = dict(info)
        if info.get(ApiField.NAME) is None and info.get(ApiField.TITLE) is not None:
            info[ApiField.NAME] = info[ApiField.TITLE]
        if info.get(ApiField.TITLE) is None and info.get(ApiField.NAME) is not None:
            info[ApiField.TITLE] = info[ApiField.NAME]
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        return MeshInfo(**res._asdict())

    @staticmethod
    def _validate_project_and_dataset_id(project_id: Optional[int], dataset_id: Optional[int]) -> None:
        """
        Ensure that exactly one of *project_id* or *dataset_id* is provided.

        :raises ValueError: If neither or both arguments are provided.
        """
        if project_id is None and dataset_id is None:
            raise ValueError("Either project_id or dataset_id must be provided.")
        if project_id is not None and dataset_id is not None:
            raise ValueError("Only one of project_id or dataset_id can be provided.")

    @staticmethod
    def _validate_mesh_name(name: str) -> None:
        """
        Validate that *name* has a supported mesh extension (``.ply``, ``.stl``, ``.obj``).

        :raises ValueError: If the extension is not allowed.
        """
        ext = get_file_ext(name).lower()
        if ext not in ALLOWED_MESH_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_MESH_EXTENSIONS))
            raise ValueError(f"Unsupported mesh extension {ext!r}. Allowed extensions: {allowed}.")

    @staticmethod
    def _reserve_unique_path(path: str, reserved_paths: set) -> str:
        """
        Return a path not already present in *reserved_paths*, adding a numeric suffix if needed.

        The chosen path is added to *reserved_paths* in place.

        :param path: Desired path.
        :type path: str
        :param reserved_paths: Set of already-reserved paths (mutated in place).
        :type reserved_paths: set
        :returns: A unique path.
        :rtype: str
        """
        candidate = path
        suffix = 0
        parsed_path = PurePosixPath(path)
        while candidate in reserved_paths:
            candidate = str(parsed_path.with_name(f"{parsed_path.stem}_{suffix:03d}{parsed_path.suffix}"))
            suffix += 1
        reserved_paths.add(candidate)
        return candidate

    def _remove_batch_api_method_name(self):
        """Return the API method name used for batch removal of mesh entities."""
        return "entities.bulk.remove"

    def _remove_batch_field_name(self):
        """Return the request field name carrying the list of IDs for batch removal."""
        return ApiField.IDS

    def get_list(
        self,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = "id",
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
        fields: Optional[List[str]] = None,
        extra_fields: Optional[List[str]] = None,
        recursive: Optional[bool] = False,
        show_disabled: Optional[bool] = False,
    ) -> List[MeshInfo]:
        """
        Get a list of mesh entities for a given dataset or project.

        Exactly one of *dataset_id* or *project_id* must be provided.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int, optional
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param filters: List of filter conditions. See the ``entities.list`` API docs.
        :type filters: List[Dict[str, str]], optional
        :param sort: Field name to sort results by. Defaults to ``"id"``.
        :type sort: str, optional
        :param sort_order: Sort direction — ``"asc"`` or ``"desc"``. Defaults to ``"asc"``.
        :type sort_order: str, optional
        :param limit: Maximum number of items to return. ``None`` returns all items.
        :type limit: int, optional
        :param fields: Explicit list of fields to include in each returned item.
            Defaults to :meth:`default_fields`.
        :type fields: List[str], optional
        :param extra_fields: Additional fields to append on top of *fields*.
        :type extra_fields: List[str], optional
        :param recursive: If ``True``, include meshes from nested datasets.
        :type recursive: bool, optional
        :param show_disabled: If ``True``, include disabled meshes.
        :type show_disabled: bool, optional
        :returns: List of :class:`MeshInfo` objects.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_infos = api.mesh.get_list(dataset_id=4)
                print(mesh_infos)
                # Output: [MeshInfo(...), MeshInfo(...)]
        """
        self._validate_project_and_dataset_id(project_id, dataset_id)
        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.DATASET_ID: dataset_id,
            ApiField.FILTER: filters or [],
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
            ApiField.FIELDS: fields or self.default_fields(),
            ApiField.RECURSIVE: recursive,
            ApiField.SHOW_DISABLED: show_disabled,
        }
        if extra_fields is not None:
            data[ApiField.EXTRA_FIELDS] = extra_fields
        return self.get_list_all_pages("entities.list", data, limit=limit)

    def get_info_by_id(self, id: int, fields: Optional[List[str]] = None, raise_error: bool = False) -> MeshInfo:
        """
        Get mesh information by ID.

        :param id: Mesh ID in Supervisely.
        :type id: int
        :param fields: Explicit list of fields to return. Defaults to :meth:`default_fields`.
        :type fields: List[str], optional
        :param raise_error: If ``True``, raise :exc:`KeyError` when the mesh is not found.
        :type raise_error: bool
        :returns: Information about the mesh, or ``None`` if not found and *raise_error* is ``False``.
        :rtype: :class:`~supervisely.api.mesh.mesh_api.MeshInfo`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_info = api.mesh.get_info_by_id(1)
                print(mesh_info)
                # Output: MeshInfo(id=1, name='scan.stl', ...)
        """
        info = self._get_info_by_id(
            id,
            "entities.info",
            {ApiField.FIELDS: fields or self.default_fields()},
        )
        if info is None and raise_error:
            raise KeyError(f"Mesh with id={id} not found in your account")
        return info

    def get_info_by_name(
        self,
        dataset_id: int,
        name: str,
        fields: Optional[List[str]] = None,
    ) -> MeshInfo:
        """
        Get mesh information by its filename within a dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Mesh filename to look up.
        :type name: str
        :param fields: Explicit list of fields to return. Defaults to :meth:`default_fields`.
        :type fields: List[str], optional
        :returns: Information about the mesh, or ``None`` if not found.
        :rtype: :class:`~supervisely.api.mesh.mesh_api.MeshInfo`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_info = api.mesh.get_info_by_name(dataset_id=4, name="scan.stl")
        """
        filters = [{ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "=", ApiField.VALUE: name}]
        return _get_single_item(self.get_list(dataset_id=dataset_id, filters=filters, fields=fields))

    def download_path(self, id: int, path: str) -> None:
        """
        Download a mesh file to a local path.

        :param id: Mesh ID in Supervisely.
        :type id: int
        :param path: Local destination path (including filename).
        :type path: str
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                api.mesh.download_path(id=1, path="/tmp/scan.stl")
        """
        response = self._api.post("entities.download", {ApiField.ID: id}, stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def upload_link(
        self,
        dataset_id: int,
        link: str,
        name: Optional[str] = None,
        meta: Optional[Dict] = None,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> MeshInfo:
        """
        Upload a single mesh from an external URL.

        If *name* is omitted, a random name with the URL's file extension is used.
        A free (non-conflicting) name is resolved automatically via :meth:`get_free_name`.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param link: Public URL pointing to the mesh file.
        :type link: str
        :param name: Desired filename (with extension). Defaults to a random name derived from *link*.
        :type name: str, optional
        :param meta: Arbitrary metadata to attach to the mesh.
        :type meta: dict, optional
        :param description: Human-readable description of the mesh.
        :type description: str, optional
        :param parent_id: ID of the parent entity, if applicable.
        :type parent_id: int, optional
        :returns: Information about the uploaded mesh.
        :rtype: :class:`~supervisely.api.mesh.mesh_api.MeshInfo`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_info = api.mesh.upload_link(
                    dataset_id=4,
                    link="https://example.com/scan.stl",
                )
        """
        if name is None:
            url_path = urlparse(link).path
            name = rand_str(10) + get_file_ext(url_path)
        name = self.get_free_name(dataset_id, name)
        return self.upload_links(
            dataset_id,
            [name],
            [link],
            metas=[meta] if meta is not None else None,
            descriptions=[description] if description is not None else None,
            parent_ids=[parent_id] if parent_id is not None else None,
        )[0]

    def upload_links(
        self,
        dataset_id: int,
        names: List[str],
        links: List[str],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
        """
        Upload multiple meshes from external URLs.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: List of filenames (with extensions) for the uploaded meshes.
        :type names: List[str]
        :param links: List of public URLs. Must be the same length as *names*.
        :type links: List[str]
        :param metas: Per-mesh metadata dictionaries. Defaults to empty dicts.
        :type metas: List[dict], optional
        :param progress_cb: Callable invoked with the number of items processed per batch.
        :type progress_cb: Callable, optional
        :param descriptions: Per-mesh human-readable descriptions.
        :type descriptions: List[str], optional
        :param parent_ids: Per-mesh parent entity IDs.
        :type parent_ids: List[int], optional
        :returns: List of :class:`MeshInfo` objects in the same order as *names*.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                infos = api.mesh.upload_links(
                    dataset_id=4,
                    names=["a.stl", "b.obj"],
                    links=["https://example.com/a.stl", "https://example.com/b.obj"],
                )
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.LINK, item),
            dataset_id,
            names,
            links,
            metas=metas,
            progress_cb=progress_cb,
            descriptions=descriptions,
            parent_ids=parent_ids,
        )

    def upload_ids(
        self,
        dataset_id: int,
        names: List[str],
        ids: List[int],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
        """
        Upload meshes from given source IDs to a dataset (server-side copy).

        Mirrors :meth:`~supervisely.api.image_api.ImageApi.upload_ids`: each source
        mesh is re-registered in the destination dataset by its ID; the binary is
        not downloaded or re-uploaded.

        :param dataset_id: Destination dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Destination mesh names with extension. Must match *ids* length.
        :type names: List[str]
        :param ids: Source mesh IDs in Supervisely.
        :type ids: List[int]
        :param metas: Per-mesh metadata dictionaries. Defaults to empty dicts.
        :type metas: List[dict], optional
        :param progress_cb: Progress callback.
        :type progress_cb: tqdm or callable, optional
        :param descriptions: Per-mesh human-readable descriptions.
        :type descriptions: List[str], optional
        :param parent_ids: Per-mesh parent entity IDs.
        :type parent_ids: List[int], optional
        :returns: List of :class:`MeshInfo` objects in the same order as *ids*.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                src_infos = api.mesh.get_list(src_dataset_id)
                names = [info.name for info in src_infos]
                ids = [info.id for info in src_infos]
                new_infos = api.mesh.upload_ids(dst_dataset_id, names, ids)
        """
        if metas is None:
            metas = [{}] * len(names)
        return self._upload_bulk_add(
            lambda item: (ApiField.ENTITY_ID, item),
            dataset_id,
            names,
            ids,
            metas=metas,
            progress_cb=progress_cb,
            descriptions=descriptions,
            parent_ids=parent_ids,
        )

    def _upload_by_team_file_ids(
        self,
        dataset_id: int,
        names: List[str],
        team_file_ids: List[int],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
        """
        Register meshes in a dataset from previously uploaded Team Files IDs.

        :param dataset_id: Destination dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Destination mesh names with extension. Must match *team_file_ids* length.
        :type names: List[str]
        :param team_file_ids: Team Files IDs of the mesh files.
        :type team_file_ids: List[int]
        :param metas: Per-mesh metadata dictionaries. Defaults to empty dicts.
        :type metas: List[dict], optional
        :param progress_cb: Progress callback.
        :type progress_cb: Callable, optional
        :param descriptions: Per-mesh human-readable descriptions.
        :type descriptions: List[str], optional
        :param parent_ids: Per-mesh parent entity IDs.
        :type parent_ids: List[int], optional
        :returns: List of :class:`MeshInfo` objects in the same order as *names*.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]
        """
        return self._upload_bulk_add(
            lambda item: (ApiField.TEAM_FILE_ID, item),
            dataset_id,
            names,
            team_file_ids,
            metas=metas,
            progress_cb=progress_cb,
            descriptions=descriptions,
            parent_ids=parent_ids,
        )

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Optional[Dict] = None,
        team_files_dir: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> MeshInfo:
        """
        Upload a single mesh from a local file.

        The file is first staged in Team Files and then registered as a mesh entity.
        See :meth:`upload_paths` for parameter details.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Filename (with extension) to use in Supervisely.
        :type name: str
        :param path: Local filesystem path to the mesh file.
        :type path: str
        :param meta: Arbitrary metadata to attach to the mesh.
        :type meta: dict, optional
        :param team_files_dir: Remote Team Files directory for staging. Defaults to
            ``/supervisely/mesh_uploads/<dataset_id>``.
        :type team_files_dir: str, optional
        :param description: Human-readable description of the mesh.
        :type description: str, optional
        :param parent_id: ID of the parent entity, if applicable.
        :type parent_id: int, optional
        :returns: Information about the uploaded mesh.
        :rtype: :class:`~supervisely.api.mesh.mesh_api.MeshInfo`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_info = api.mesh.upload_path(
                    dataset_id=4,
                    name="scan.stl",
                    path="/local/data/scan.stl",
                )
        """
        return self.upload_paths(
            dataset_id,
            [name],
            [path],
            metas=[meta] if meta is not None else None,
            team_files_dir=team_files_dir,
            descriptions=[description] if description is not None else None,
            parent_ids=[parent_id] if parent_id is not None else None,
        )[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        metas: Optional[List[Dict]] = None,
        team_files_dir: Optional[str] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
        """
        Upload multiple meshes from local files.

        Each file is first staged under *team_files_dir* in Team Files using
        :meth:`~supervisely.api.file_api.FileApi.upload_bulk`, and then
        registered as a mesh entity via Team Files IDs.  Only ``.ply``,
        ``.stl``, and ``.obj`` extensions are accepted for both *names* and
        *paths*.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Filenames (with extensions) to use in Supervisely.
            Must be the same length as *paths*.
        :type names: List[str]
        :param paths: Local filesystem paths to the mesh files.
            Must be the same length as *names*.
        :type paths: List[str]
        :param progress_cb: Callable invoked with the number of bytes/items processed
            during the Team Files upload stage.
        :type progress_cb: Union[tqdm, Callable], optional
        :param metas: Per-mesh metadata dictionaries. Defaults to empty dicts.
        :type metas: List[dict], optional
        :param team_files_dir: Remote Team Files directory used for staging.
            Defaults to ``/supervisely/mesh_uploads/<dataset_id>``.
        :type team_files_dir: str, optional
        :param descriptions: Per-mesh human-readable descriptions.
        :type descriptions: List[str], optional
        :param parent_ids: Per-mesh parent entity IDs.
        :type parent_ids: List[int], optional
        :raises RuntimeError: If *names* and *paths* have different lengths.
        :raises ValueError: If any name or path has an unsupported extension.
        :raises FileNotFoundError: If any path does not point to an existing file.
        :returns: List of :class:`MeshInfo` objects in the same order as *names*.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                infos = api.mesh.upload_paths(
                    dataset_id=4,
                    names=["a.stl", "b.obj"],
                    paths=["/local/a.stl", "/local/b.obj"],
                )
        """
        if len(names) != len(paths):
            raise RuntimeError('Can not match "names" and "paths" lists, len(names) != len(paths)')
        if len(names) == 0:
            return []
        for name, path in zip(names, paths):
            self._validate_mesh_name(name)
            self._validate_mesh_name(path)
            name_ext = get_file_ext(name).lower()
            path_ext = get_file_ext(path).lower()
            if name_ext != path_ext:
                raise ValueError(
                    f"The name extension '{name_ext}' does not match the file extension '{path_ext}'"
                )
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        dataset_info = self._api.dataset.get_info_by_id(dataset_id)
        team_id = dataset_info.team_id
        team_files_dir = team_files_dir or f"/supervisely/mesh_uploads/{dataset_id}"

        dst_paths = []
        reserved_dst_paths = set()
        for name in names:
            remote_path = f"{team_files_dir.rstrip('/')}/{name}"
            free_path = self._api.file.get_free_name(team_id, remote_path)
            dst_paths.append(self._reserve_unique_path(free_path, reserved_dst_paths))

        file_infos = self._api.file.upload_bulk(team_id, paths, dst_paths, progress_cb=progress_cb)
        team_file_ids = [file_info.id for file_info in file_infos]
        return self._upload_by_team_file_ids(
            dataset_id,
            names,
            team_file_ids,
            metas=metas,
            progress_cb=None,
            descriptions=descriptions,
            parent_ids=parent_ids,
        )

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id: int,
        names: List[str],
        items: List,
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
        """
        Register meshes in a dataset in batches via the ``entities.bulk.add`` method.

        *func_item_to_kv* maps each item in *items* to the ``(field, value)`` pair that
        identifies the source (e.g. a link, an entity ID, or a Team Files ID).

        :param func_item_to_kv: Callable mapping an item to its ``(ApiField, value)`` pair.
        :type func_item_to_kv: Callable
        :param dataset_id: Destination dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Destination mesh names with extension. Must match *items* length.
        :type names: List[str]
        :param items: Source items resolved by *func_item_to_kv*.
        :type items: list
        :param metas: Per-mesh metadata dictionaries. Defaults to empty dicts.
        :type metas: List[dict], optional
        :param progress_cb: Progress callback invoked with the batch size.
        :type progress_cb: Callable, optional
        :param descriptions: Per-mesh human-readable descriptions.
        :type descriptions: List[str], optional
        :param parent_ids: Per-mesh parent entity IDs.
        :type parent_ids: List[int], optional
        :raises RuntimeError: If the input lists have mismatched lengths.
        :returns: List of :class:`MeshInfo` objects in the same order as *names*.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]
        """
        if len(names) == 0:
            return []
        if len(names) != len(items):
            raise RuntimeError('Can not match "names" and "items" lists, len(names) != len(items)')
        if metas is None:
            metas = [{}] * len(items)
        if descriptions is None:
            descriptions = [None] * len(items)
        if parent_ids is None:
            parent_ids = [None] * len(items)
        if not (len(names) == len(metas) == len(descriptions) == len(parent_ids)):
            raise RuntimeError("names, metas, descriptions and parent_ids must have the same length")

        results = []
        for batch in batched(list(zip(names, items, metas, descriptions, parent_ids))):
            entities = []
            for name, item, meta, description, parent_id in batch:
                self._validate_mesh_name(name)
                item_field, item_value = func_item_to_kv(item)
                entity = {
                    ApiField.NAME: name,
                    item_field: item_value,
                    ApiField.META: meta if meta is not None else {},
                }
                if description is not None:
                    entity[ApiField.DESCRIPTION] = description
                if parent_id is not None:
                    entity[ApiField.PARENT_ID] = parent_id
                entities.append(entity)

            response = self._api.post(
                "entities.bulk.add",
                {ApiField.DATASET_ID: dataset_id, ApiField.ENTITIES: entities},
            )
            if progress_cb is not None:
                progress_cb(len(entities))
            results.extend([self._convert_json_info(info) for info in response.json()])

        name_to_result = {mesh_info.name: mesh_info for mesh_info in results}
        return [name_to_result.get(name, results[idx]) for idx, name in enumerate(names)]

    def copy_batch(
        self,
        dst_dataset_id: int,
        ids: List[int],
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[MeshInfo]:
        """
        Copy meshes with given IDs to a dataset.

        :param dst_dataset_id: Destination Dataset ID in Supervisely.
        :type dst_dataset_id: int
        :param ids: Mesh IDs to copy.
        :type ids: List[int]
        :param change_name_if_conflict: Add a suffix when a mesh with the same name already exists
            in the destination dataset. When False and a name collision is detected, raises ValueError.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: Copy mesh annotations to the destination dataset.
        :type with_annotations: bool, optional
        :param progress_cb: Progress callback invoked once per copied mesh.
        :type progress_cb: tqdm or callable, optional
        :raises TypeError: If *ids* is not a list.
        :raises ValueError: If meshes from more than one source dataset are given, or if name
            conflicts exist and *change_name_if_conflict* is False.
        :returns: List of new :class:`MeshInfo` objects.
        :rtype: List[:class:`~supervisely.api.mesh.mesh_api.MeshInfo`]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()

                mesh_infos = api.mesh.get_list(src_dataset_id)
                mesh_ids = [info.id for info in mesh_infos]
                new_infos = api.mesh.copy_batch(dst_dataset_id, mesh_ids, with_annotations=True)
        """
        if type(ids) is not list:
            raise TypeError(
                "ids parameter has type {!r}, but has to be of type {!r}".format(type(ids), list)
            )

        if len(ids) == 0:
            return []

        ids_info = [self.get_info_by_id(mesh_id) for mesh_id in ids]
        if len(set(info.dataset_id for info in ids_info)) > 1:
            raise ValueError("Mesh ids have to be from the same dataset")

        existing_meshes = self.get_list(dst_dataset_id)
        existing_names = {mesh.name for mesh in existing_meshes}

        if change_name_if_conflict:
            new_names = [
                generate_free_name(existing_names, info.name, with_ext=True, extend_used_names=True)
                for info in ids_info
            ]
        else:
            new_names = [info.name for info in ids_info]
            names_intersection = existing_names.intersection(set(new_names))
            if len(names_intersection) != 0:
                raise ValueError(
                    "Meshes with the same names already exist in destination dataset. "
                    'Please, use argument "change_name_if_conflict=True" to automatically resolve '
                    "names intersection"
                )

        new_meshes = self.upload_ids(dst_dataset_id, new_names, ids, progress_cb=progress_cb)
        new_ids = [mesh.id for mesh in new_meshes]

        if with_annotations:
            src_project_id = self._api.dataset.get_info_by_id(ids_info[0].dataset_id).project_id
            dst_project_id = self._api.dataset.get_info_by_id(dst_dataset_id).project_id
            self._api.project.merge_metas(src_project_id, dst_project_id)
            self._api.mesh.annotation.copy_batch(ids, new_ids)

        return new_meshes
