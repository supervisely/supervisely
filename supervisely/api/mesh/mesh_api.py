# coding: utf-8
from __future__ import annotations

import os
from typing import Callable, Dict, List, NamedTuple, Optional, Union
from urllib.parse import urlparse

from tqdm import tqdm

from supervisely._utils import batched, rand_str
from supervisely.api.mesh.mesh_annotation_api import MeshAnnotationAPI
from supervisely.api.mesh.mesh_figure_api import MeshFigureApi
from supervisely.api.mesh.mesh_object_api import MeshObjectApi
from supervisely.api.mesh.mesh_tag_api import MeshTagApi
from supervisely.api.module_api import ApiField, ModuleApi, _get_single_item
from supervisely.io.fs import ensure_base_path, get_file_ext


ALLOWED_MESH_EXTENSIONS = {".ply", ".stl", ".obj", ".sly"}


class MeshInfo(NamedTuple):
    """Information about a mesh entity."""

    id: int
    name: str
    title: str
    description: str
    parent_id: int
    workspace_id: int
    project_id: int
    dataset_id: int
    path_original: str
    full_storage_url: str
    link: str
    meta: dict
    file_meta: dict
    frame: int
    size: int
    custom_data: dict
    objects_count: int
    created_by_id: int
    created_at: str
    updated_at: str


class MeshApi(ModuleApi):
    """API for working with mesh entities."""

    def __init__(self, api):
        super().__init__(api)
        self.annotation = MeshAnnotationAPI(api)
        self.object = MeshObjectApi(api)
        self.figure = MeshFigureApi(api)
        self.tag = MeshTagApi(api)

    @staticmethod
    def info_sequence():
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
            ApiField.CREATED_BY_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        return "MeshInfo"

    @staticmethod
    def default_fields() -> List[str]:
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
            "createdBy",
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
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
        if project_id is None and dataset_id is None:
            raise ValueError("Either project_id or dataset_id must be provided.")
        if project_id is not None and dataset_id is not None:
            raise ValueError("Only one of project_id or dataset_id can be provided.")

    @staticmethod
    def _validate_mesh_name(name: str) -> None:
        ext = get_file_ext(name).lower()
        if ext not in ALLOWED_MESH_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_MESH_EXTENSIONS))
            raise ValueError(f"Unsupported mesh extension {ext!r}. Allowed extensions: {allowed}.")

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

    def get_info_by_id(self, id: int, fields: Optional[List[str]] = None) -> MeshInfo:
        return self._get_info_by_id(
            id,
            "entities.info",
            {ApiField.FIELDS: fields or self.default_fields()},
        )

    def _get_json_info_by_id(self, id: int, fields: Optional[List[str]] = None) -> Dict:
        response = self._get_response_by_id(
            id,
            "entities.info",
            ApiField.ID,
            {ApiField.FIELDS: fields or self.default_fields()},
        )
        return response.json() if response is not None else None

    def get_info_by_name(
        self,
        dataset_id: int,
        name: str,
        fields: Optional[List[str]] = None,
    ) -> MeshInfo:
        filters = [{ApiField.FIELD: ApiField.NAME, ApiField.OPERATOR: "=", ApiField.VALUE: name}]
        return _get_single_item(self.get_list(dataset_id=dataset_id, filters=filters, fields=fields))

    def download_path(self, id: int, path: str) -> None:
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

    def upload_team_file_id(
        self,
        dataset_id: int,
        name: str,
        team_file_id: int,
        meta: Optional[Dict] = None,
        description: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> MeshInfo:
        return self.upload_team_file_ids(
            dataset_id,
            [name],
            [team_file_id],
            metas=[meta] if meta is not None else None,
            descriptions=[description] if description is not None else None,
            parent_ids=[parent_id] if parent_id is not None else None,
        )[0]

    def upload_team_file_ids(
        self,
        dataset_id: int,
        names: List[str],
        team_file_ids: List[int],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
        descriptions: Optional[List[str]] = None,
        parent_ids: Optional[List[int]] = None,
    ) -> List[MeshInfo]:
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
        if len(names) != len(paths):
            raise RuntimeError('Can not match "names" and "paths" lists, len(names) != len(paths)')
        if len(names) == 0:
            return []
        for name, path in zip(names, paths):
            self._validate_mesh_name(name)
            self._validate_mesh_name(path)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        dataset_info = self._api.dataset.get_info_by_id(dataset_id)
        team_id = dataset_info.team_id
        team_files_dir = team_files_dir or f"/supervisely/mesh_uploads/{dataset_id}"

        dst_paths = []
        for name in names:
            remote_path = f"{team_files_dir.rstrip('/')}/{name}"
            dst_paths.append(self._api.file.get_free_name(team_id, remote_path))

        file_infos = self._api.file.upload_bulk(team_id, paths, dst_paths, progress_cb=progress_cb)
        team_file_ids = [file_info.id for file_info in file_infos]
        return self.upload_team_file_ids(
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
