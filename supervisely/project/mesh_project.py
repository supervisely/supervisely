# coding: utf-8
from __future__ import annotations

import os
import re
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.api import Api
from supervisely.api.mesh.mesh_api import ALLOWED_MESH_EXTENSIONS, MeshInfo
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.fs import (
    clean_dir,
    dir_exists,
    ensure_base_path,
    file_exists,
    get_file_ext,
    list_files,
    mkdir,
    remove_dir,
    silent_remove,
    touch,
)
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.mesh_annotation.constants import FIGURES, KEY, LABELS, OBJECTS, TAGS
from supervisely.mesh_annotation.mesh_annotation import MeshAnnotation
from supervisely.mesh_annotation.mesh_indices import (
    MESH_INDEX_FIELDS,
    decode_mesh_indices,
    encode_mesh_indices,
)
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project import read_single_project as read_project_wrapper
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap


ANNOTATION_FILE_NAME = "annotation.json"
GEOMETRIES_DIR_NAME = "geometries"


class MeshItemPaths(NamedTuple):
    """Paths to a mesh item, its annotation directory, and geometry sidecars."""

    mesh_path: str
    ann_dir: str
    ann_path: str
    geometries_dir: str


class MeshItemInfo(NamedTuple):
    """Basic info about a mesh item and where its files are stored on disk."""

    dataset_name: str
    name: str
    mesh_path: str
    ann_dir: str
    ann_path: str
    geometries_dir: str


class MeshDataset(Dataset):
    """A dataset directory for mesh items inside a local Supervisely mesh project."""

    item_dir_name = "meshes"
    ann_dir_name = "annotations"
    item_info_dir_name = "mesh_info"
    seg_dir_name = None

    annotation_class = MeshAnnotation
    item_info_class = MeshInfo

    @property
    def mesh_dir(self) -> str:
        return self.item_dir

    @property
    def mesh_info_dir(self) -> str:
        return self.item_info_dir

    @property
    def img_dir(self) -> str:
        raise NotImplementedError(
            f"Property 'img_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def img_info_dir(self):
        raise NotImplementedError(
            f"Property 'img_info_dir' is not supported for {type(self).__name__} object."
        )

    @property
    def seg_dir(self):
        raise NotImplementedError(
            f"Property 'seg_dir' is not supported for {type(self).__name__} object."
        )

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        return get_file_ext(path).lower() in ALLOWED_MESH_EXTENSIONS

    def _create(self):
        mkdir(self.ann_dir)
        mkdir(self.item_dir)

    def _read(self):
        if not dir_exists(self.item_dir):
            raise FileNotFoundError("Item directory not found: {!r}".format(self.item_dir))
        if not dir_exists(self.ann_dir):
            raise FileNotFoundError("Annotation directory not found: {!r}".format(self.ann_dir))

        mesh_paths = list_files(self.item_dir, filter_fn=self._has_valid_ext)
        mesh_names = [os.path.basename(path) for path in mesh_paths]

        ann_names = []
        for entry in os.scandir(self.ann_dir):
            if entry.is_dir() and file_exists(os.path.join(entry.path, ANNOTATION_FILE_NAME)):
                ann_names.append(entry.name)

        if len(mesh_names) == 0 and len(ann_names) == 0:
            logger.debug(f"Dataset '{self.name}' is empty")

        if len(mesh_names) == 0:
            mesh_names = ann_names

        effective_ann_names = set()
        for mesh_name in mesh_names:
            ann_path = self._ann_path_by_name(mesh_name)
            if not file_exists(ann_path):
                raise RuntimeError(
                    "Item {!r} in dataset {!r} does not have a corresponding annotation file.".format(
                        mesh_name, self.name
                    )
                )
            if mesh_name in effective_ann_names:
                raise RuntimeError(
                    "Annotation directory {!r} in dataset {!r} matches two different mesh files.".format(
                        mesh_name, self.name
                    )
                )
            effective_ann_names.add(mesh_name)
            self._item_to_ann[mesh_name] = mesh_name

    def _get_empty_annotaion(self, item_name):
        return self.annotation_class()

    def _add_item_file(self, item_name, item_path, _validate_item=True, _use_hardlink=False):
        if item_path is not None and not self._has_valid_ext(item_path):
            raise RuntimeError("Item path {!r} has unsupported extension.".format(item_path))
        super()._add_item_file(
            item_name,
            item_path,
            _validate_item=False,
            _use_hardlink=_use_hardlink,
        )

    def _validate_added_item_or_die(self, item_path):
        if not self._has_valid_ext(item_path):
            raise RuntimeError("Item path {!r} has unsupported extension.".format(item_path))

    def _add_ann_by_type(self, item_name, ann):
        self._item_to_ann[item_name] = item_name
        if ann is None:
            self.set_ann(item_name, self._get_empty_annotaion(item_name))
        elif isinstance(ann, self.annotation_class):
            self.set_ann(item_name, ann)
        elif type(ann) is str:
            self.set_ann_file(item_name, ann)
        elif type(ann) is dict:
            self.set_ann_dict(item_name, ann)
        else:
            raise TypeError("Unsupported type {!r} for ann argument".format(type(ann)))

    def get_img_path(self, item_name: str) -> str:
        raise NotImplementedError(
            f"Method 'get_img_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_img_info_path(self, img_name: str) -> str:
        raise NotImplementedError(
            f"Method 'get_img_info_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_image_info(self, item_name: str) -> None:
        raise NotImplementedError(
            f"Method 'get_image_info(item_name)' is not supported for {type(self).__name__} object."
        )

    def get_seg_path(self, item_name: str) -> str:
        raise NotImplementedError(
            f"Method 'get_seg_path(item_name)' is not supported for {type(self).__name__} object."
        )

    def add_item_np(self, item_name, img, ann=None, img_info=None):
        raise NotImplementedError(
            f"Method 'add_item_np()' is not supported for {type(self).__name__} object."
        )

    def add_item_raw_bytes(self, item_name, item_raw_bytes, ann=None, img_info=None):
        raise NotImplementedError(
            f"Method 'add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_item_raw_bytes(self, item_name, item_raw_bytes):
        raise NotImplementedError(
            f"Method '_add_item_raw_bytes()' is not supported for {type(self).__name__} object."
        )

    def _add_img_np(self, item_name, img):
        raise NotImplementedError(
            f"Method '_add_img_np()' is not supported for {type(self).__name__} object."
        )

    def get_mesh_path(self, item_name: str) -> str:
        return self.get_item_path(item_name)

    def get_mesh_info(self, item_name: str) -> MeshInfo:
        return self.get_item_info(item_name)

    def get_mesh_info_path(self, item_name: str) -> str:
        return self.get_item_info_path(item_name)

    def get_item_info_path(self, item_name: str) -> str:
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        return os.path.join(self.item_info_dir, item_name + ".json")

    def _ann_dir_by_name(self, item_name: str) -> str:
        return os.path.join(self.ann_dir, item_name)

    def _ann_path_by_name(self, item_name: str) -> str:
        return os.path.join(self._ann_dir_by_name(item_name), ANNOTATION_FILE_NAME)

    def _geometries_dir_by_name(self, item_name: str) -> str:
        return os.path.join(self._ann_dir_by_name(item_name), GEOMETRIES_DIR_NAME)

    def get_annotation_dir(self, item_name: str) -> str:
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        return self._ann_dir_by_name(item_name)

    def get_geometry_dir(self, item_name: str) -> str:
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        return self._geometries_dir_by_name(item_name)

    def get_ann_path(self, item_name: str) -> str:
        if not self.item_exists(item_name):
            raise RuntimeError("Item {} not found in the project.".format(item_name))
        return self._ann_path_by_name(item_name)

    def get_ann_json(self, item_name: str, decode_geometries: bool = True) -> Dict:
        ann_json = load_json_file(self.get_ann_path(item_name))
        if decode_geometries:
            ann_json = self._decode_geometry_sidecars(item_name, ann_json)
        return ann_json

    def get_ann(
        self, item_name, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> MeshAnnotation:
        return self.annotation_class.from_json(self.get_ann_json(item_name), project_meta, key_id_map)

    def set_ann(
        self, item_name: str, ann: MeshAnnotation, key_id_map: Optional[KeyIdMap] = None
    ) -> None:
        if not isinstance(ann, self.annotation_class):
            raise TypeError(
                f"Type of 'ann' should be {self.annotation_class.__name__}, not a {type(ann).__name__}"
            )
        self.set_ann_dict(item_name, ann.to_json(key_id_map))

    def set_ann_file(self, item_name: str, ann_path: str) -> None:
        if type(ann_path) is not str:
            raise TypeError("Annotation path should be a string, not a {}".format(type(ann_path)))
        ann_json = self._decode_geometry_sidecars_from_dir(
            os.path.dirname(ann_path), load_json_file(ann_path)
        )
        self.set_ann_dict(item_name, ann_json)

    def set_ann_dict(self, item_name: str, ann: Dict) -> None:
        if type(ann) is not dict:
            raise TypeError("Ann should be a dict, not a {}".format(type(ann)))
        ann_dir = self._ann_dir_by_name(item_name)
        geometries_dir = self._geometries_dir_by_name(item_name)
        mkdir(ann_dir)
        mkdir(geometries_dir)
        clean_dir(geometries_dir)

        stored_ann = self._encode_geometry_sidecars(item_name, ann)
        dst_ann_path = self._ann_path_by_name(item_name)
        ensure_base_path(dst_ann_path)
        dump_json_file(stored_ann, dst_ann_path, indent=4)

    def get_item_paths(self, item_name: str) -> MeshItemPaths:
        return MeshItemPaths(
            mesh_path=self.get_mesh_path(item_name),
            ann_dir=self.get_annotation_dir(item_name),
            ann_path=self.get_ann_path(item_name),
            geometries_dir=self.get_geometry_dir(item_name),
        )

    def items(self) -> Generator[Tuple[str, str, str], None, None]:
        for item_name in self._item_to_ann.keys():
            yield item_name, self.get_mesh_path(item_name), self.get_ann_path(item_name)

    def delete_item(self, item_name: str) -> bool:
        if self.item_exists(item_name):
            paths = self.get_item_paths(item_name)
            info_path = self.get_mesh_info_path(item_name)
            silent_remove(paths.mesh_path)
            silent_remove(info_path)
            if dir_exists(paths.ann_dir):
                remove_dir(paths.ann_dir)
            self._item_to_ann.pop(item_name)
            return True
        return False

    def add_item_file(
        self,
        item_name: str,
        item_path: str,
        ann: Optional[Union[MeshAnnotation, Dict, str]] = None,
        _validate_item: Optional[bool] = True,
        _use_hardlink: Optional[bool] = False,
        item_info: Optional[Union[MeshInfo, Dict, str]] = None,
    ) -> None:
        return super().add_item_file(
            item_name=item_name,
            item_path=item_path,
            ann=ann,
            _validate_item=_validate_item,
            _use_hardlink=_use_hardlink,
            item_info=item_info,
        )

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = MeshProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta

        class_items = {obj_class.name: 0 for obj_class in project_meta.obj_classes}
        class_objects = {obj_class.name: 0 for obj_class in project_meta.obj_classes}
        class_figures = {obj_class.name: 0 for obj_class in project_meta.obj_classes}

        for item_name in self:
            ann_json = self.get_ann_json(item_name)
            item_classes = set()
            for label_json in ann_json.get(LABELS, []):
                class_name = label_json.get("classTitle")
                if class_name in class_figures:
                    class_objects[class_name] += 1
                    class_figures[class_name] += 1
                    item_classes.add(class_name)

            for class_name in item_classes:
                class_items[class_name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            result["figures_count"] = class_figures
        return result

    def _encode_geometry_sidecars(self, item_name: str, ann_json: Dict) -> Dict:
        stored_ann = deepcopy(ann_json)
        ann_dir = self._ann_dir_by_name(item_name)

        for label_idx, label_json in enumerate(stored_ann.get(LABELS, [])):
            if not isinstance(label_json, dict):
                continue
            geometry = label_json.get(ApiField.GEOMETRY)
            if not isinstance(geometry, dict):
                continue
            figure_ref = self._figure_geometry_ref(label_json, label_idx)
            for field_name in MESH_INDEX_FIELDS:
                value = geometry.get(field_name)
                if not self._is_indices_sequence(value):
                    continue
                filename = f"{figure_ref}.{field_name}.bin"
                rel_path = "/".join([GEOMETRIES_DIR_NAME, filename])
                abs_path = os.path.join(ann_dir, *rel_path.split("/"))
                ensure_base_path(abs_path)
                with open(abs_path, "wb") as geometry_file:
                    geometry_file.write(encode_mesh_indices(value))
                geometry[field_name] = None
                geometry[self._geometry_path_key(field_name)] = rel_path
        return stored_ann

    def _decode_geometry_sidecars(self, item_name: str, ann_json: Dict) -> Dict:
        return self._decode_geometry_sidecars_from_dir(self._ann_dir_by_name(item_name), ann_json)

    @classmethod
    def _decode_geometry_sidecars_from_dir(cls, ann_dir: str, ann_json: Dict) -> Dict:
        restored_ann = deepcopy(ann_json)

        for label_json in restored_ann.get(LABELS, []):
            if not isinstance(label_json, dict):
                continue
            geometry = label_json.get(ApiField.GEOMETRY)
            if not isinstance(geometry, dict):
                continue
            for field_name in MESH_INDEX_FIELDS:
                path_key = cls._geometry_path_key(field_name)
                rel_path = geometry.get(path_key)
                if rel_path is None:
                    continue
                if geometry.get(field_name) is None:
                    abs_path = cls._resolve_geometry_path(ann_dir, rel_path)
                    with open(abs_path, "rb") as geometry_file:
                        geometry[field_name] = decode_mesh_indices(geometry_file.read())
                geometry.pop(path_key, None)
        return restored_ann

    @staticmethod
    def _geometry_path_key(field_name: str) -> str:
        return f"{field_name}Path"

    @staticmethod
    def _figure_geometry_ref(figure_json: Dict, figure_idx: int) -> str:
        value = figure_json.get(KEY) or figure_json.get(ApiField.ID) or f"figure_{figure_idx:06d}"
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
        value = value.strip("._")
        return value or f"figure_{figure_idx:06d}"

    @staticmethod
    def _is_indices_sequence(value: Any) -> bool:
        if isinstance(value, (str, bytes, bytearray)):
            return False
        if not isinstance(value, (list, tuple)):
            return False
        return all(isinstance(item, int) for item in value)

    @staticmethod
    def _resolve_geometry_path(ann_dir: str, rel_path: str) -> str:
        abs_base = os.path.abspath(ann_dir)
        abs_path = os.path.abspath(os.path.normpath(os.path.join(ann_dir, rel_path)))
        if abs_path != abs_base and not abs_path.startswith(abs_base + os.path.sep):
            raise RuntimeError("Geometry path {!r} points outside annotation directory.".format(rel_path))
        return abs_path


class MeshProject(Project):
    """A local Supervisely project for mesh data."""

    dataset_class = MeshDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = MeshDataset

    def __init__(self, directory: str, mode: OpenMode):
        self._key_id_map: KeyIdMap = None
        super().__init__(directory, mode)

    @classmethod
    def read_single(cls, dir) -> "MeshProject":
        return read_project_wrapper(dir, cls)

    @property
    def type(self) -> str:
        return ProjectType.MESHES.value

    def set_meta(self, new_meta: ProjectMeta) -> None:
        if new_meta.project_type != self.type:
            meta_json = new_meta.to_json()
            meta_json["projectType"] = self.type
            new_meta = ProjectMeta.from_json(meta_json)
        super().set_meta(new_meta)

    @property
    def key_id_map(self) -> KeyIdMap:
        return self._key_id_map

    def set_key_id_map(self, new_map: KeyIdMap):
        self._key_id_map = new_map
        self._key_id_map.dump_json(self._get_key_id_map_path())

    def _get_key_id_map_path(self):
        return os.path.join(self.directory, "key_id_map.json")

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super().get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_meshes: Optional[bool] = True,
        download_meshes_info: Optional[bool] = False,
        batch_size: Optional[int] = 10,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        **kwargs,
    ) -> "MeshProject":
        return download_mesh_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_meshes=download_meshes,
            download_meshes_info=download_meshes_info,
            batch_size=batch_size,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload(
        directory: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Tuple[int, str]:
        return upload_mesh_project(
            directory=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    async def download_async(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'download_async()' is not supported for MeshProject class now."
        )

    @staticmethod
    def download_bin(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'download_bin()' is not supported for MeshProject class."
        )

    @staticmethod
    def upload_bin(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'upload_bin()' is not supported for MeshProject class."
        )

    @staticmethod
    def build_snapshot(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'build_snapshot()' is not supported for MeshProject class."
        )

    @staticmethod
    def restore_snapshot(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'restore_snapshot()' is not supported for MeshProject class."
        )


def download_mesh_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_meshes: Optional[bool] = True,
    download_meshes_info: Optional[bool] = False,
    batch_size: Optional[int] = 10,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> MeshProject:
    key_id_map = KeyIdMap()
    project_fs = MeshProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    filter_fn = lambda ds: True
    if dataset_ids is not None:
        filter_fn = lambda ds: ds.id in dataset_ids

    for parents, dataset in api.dataset.tree(project_id):
        if not filter_fn(dataset):
            continue
        dataset_path = None
        if parents:
            dataset_path = "/datasets/".join(parents + [dataset.name])
        dataset_fs: MeshDataset = project_fs.create_dataset(
            ds_name=dataset.name, ds_path=dataset_path
        )
        meshes = api.mesh.get_list(dataset.id)

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Downloading meshes from: {!r}".format(dataset.name),
                total=len(meshes),
            )

        for batch in batched(meshes, batch_size=batch_size):
            mesh_ids = [mesh_info.id for mesh_info in batch]
            ann_jsons = api.mesh.annotation.download_bulk(dataset.id, mesh_ids)

            for mesh_id, mesh_info, ann_json in zip(mesh_ids, batch, ann_jsons):
                mesh_name = mesh_info.name
                mesh_file_path = dataset_fs.generate_item_path(mesh_name)
                if download_meshes:
                    try:
                        api.mesh.download_path(mesh_id, mesh_file_path)
                    except Exception as e:
                        logger.info(
                            "INFO FOR DEBUGGING",
                            extra={
                                "project_id": project_id,
                                "dataset_id": dataset.id,
                                "mesh_id": mesh_id,
                                "mesh_name": mesh_name,
                                "mesh_file_path": mesh_file_path,
                            },
                        )
                        raise e
                else:
                    touch(mesh_file_path)
                    mesh_file_path = None

                _update_key_id_map_from_annotation(key_id_map, mesh_id, ann_json)
                item_info = mesh_info._asdict() if download_meshes_info else None
                dataset_fs.add_item_file(
                    mesh_name,
                    mesh_file_path,
                    ann=ann_json,
                    _validate_item=False,
                    item_info=item_info,
                )
                if progress_cb is not None:
                    _update_progress(progress_cb, 1)

            if log_progress:
                ds_progress(len(batch))

    project_fs.set_key_id_map(key_id_map)
    return project_fs


def upload_mesh_project(
    directory: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    project_fs = MeshProject.read_single(directory)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.MESHES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    if progress_cb is not None:
        log_progress = False

    key_id_map = KeyIdMap()
    name_to_dsinfo = {}
    for dataset_fs in sorted(project_fs, key=lambda ds: len(ds.parents)):
        dataset_fs: MeshDataset
        parent_name = dataset_fs.name[: -len(dataset_fs.short_name)].rstrip("/")
        parent_info = name_to_dsinfo.get(parent_name)
        parent_id = None
        if parent_info is not None:
            parent_id = parent_info.id
        dataset = api.dataset.create(
            project.id, dataset_fs.short_name, change_name_if_conflict=True, parent_id=parent_id
        )
        name_to_dsinfo[dataset_fs.name] = dataset

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Uploading meshes to {!r}".format(dataset.name),
                total=len(dataset_fs),
            )

        for item_names in batched(list(dataset_fs), batch_size=10):
            item_paths = [dataset_fs.get_mesh_path(item_name) for item_name in item_names]
            mesh_infos = api.mesh.upload_paths(dataset.id, item_names, item_paths)
            mesh_ids = [mesh_info.id for mesh_info in mesh_infos]
            for mesh_id, item_name in zip(mesh_ids, item_names):
                ann = dataset_fs.get_ann(item_name, project_fs.meta)
                api.mesh.annotation.append(mesh_id, ann, key_id_map)
            if ds_progress is not None:
                _update_progress(ds_progress, len(item_names))

    return project.id, project.name


def _update_key_id_map_from_annotation(key_id_map: KeyIdMap, mesh_id: int, ann_json: Dict) -> None:
    if not isinstance(ann_json, dict):
        return

    _add_key_id(key_id_map.add_video, ann_json.get(KEY), mesh_id)

    for label_json in ann_json.get(LABELS, []):
        if isinstance(label_json, dict):
            _add_key_id(key_id_map.add_figure, label_json.get(KEY), label_json.get(ApiField.ID))
            _update_key_id_map_from_tags(key_id_map, label_json.get(TAGS, []))

    _update_key_id_map_from_tags(key_id_map, ann_json.get(TAGS, []))


def _update_key_id_map_from_tags(key_id_map: KeyIdMap, tags_json: List[Dict]) -> None:
    for tag_json in tags_json or []:
        if isinstance(tag_json, dict):
            _add_key_id(key_id_map.add_tag, tag_json.get(KEY), tag_json.get(ApiField.ID))


def _add_key_id(add_fn: Callable, key, id) -> None:
    if key is None:
        return
    try:
        add_fn(uuid.UUID(str(key)), id)
    except Exception:
        pass


def _update_progress(progress_cb, value: int) -> None:
    if hasattr(progress_cb, "update") and callable(getattr(progress_cb, "update")):
        progress_cb.update(value)
    else:
        progress_cb(value)
