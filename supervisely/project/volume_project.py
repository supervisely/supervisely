# coding: utf-8
import io
import json
import os
import re
import struct
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
from tqdm import tqdm

import supervisely as sly
import supervisely.volume_annotation.constants as volume_constants
from supervisely._utils import batched
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.api.volume.volume_api import VolumeInfo
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import change_directory_at_index, touch
from supervisely.project.project import OpenMode
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.project.versioning.common import (
    DEFAULT_VOLUME_SCHEMA_VERSION,
    get_volume_snapshot_schema,
)
from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress, tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume import stl_converter
from supervisely.volume import volume as sly_volume
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_figure import VolumeFigure

VolumeItemPaths = namedtuple("VolumeItemPaths", ["volume_path", "ann_path"])


class VolumeDataset(VideoDataset):
    item_dir_name = "volume"
    interpolation_dir = "interpolation"
    interpolation_dir_name = interpolation_dir
    mask_dir = "mask"
    mask_dir_name = mask_dir
    annotation_class = VolumeAnnotation
    item_module = sly_volume
    paths_tuple = VolumeItemPaths

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return sly_volume.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        path = item_name
        _, volume_meta = sly_volume.read_nrrd_serie_volume(path)
        return self.annotation_class(volume_meta)

    def get_interpolation_dir(self, item_name):
        return os.path.join(self.directory, self.interpolation_dir, item_name)

    def get_interpolation_path(self, item_name, figure):
        return os.path.join(self.get_interpolation_dir(item_name), figure.key().hex + ".stl")

    def get_mask_dir(self, item_name):
        return os.path.join(self.directory, self.mask_dir, item_name)

    def get_mask_path(self, item_name, figure):
        return os.path.join(self.get_mask_dir(item_name), figure.key().hex + ".nrrd")

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = VolumeProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        for item_name in self:
            item_ann = self.get_ann(item_name, project_meta)
            item_class = {}
            for ann_obj in item_ann.objects:
                class_objects[ann_obj.obj_class.name] += 1
            for volume_figure in item_ann.figures:
                class_figures[volume_figure.parent_object.obj_class.name] += 1
                item_class[volume_figure.parent_object.obj_class.name] = True
            for obj_class in project_meta.obj_classes:
                if obj_class.name in item_class.keys():
                    class_items[obj_class.name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            result["figures_count"] = class_figures
        return result


class VolumeProject(VideoProject):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset

    _SERIALIZATION_MAGIC = b"SLYVOLPAR"
    _SERIALIZATION_VERSION = 1
    _SECTION_PROJECT_INFO = 1
    _SECTION_PROJECT_META = 2
    _SECTION_DATASETS = 3
    _SECTION_VOLUMES = 4
    _SECTION_ANNOTATIONS = 5

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(VolumeProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @property
    def type(self) -> str:
        """
        Project type.

        :return: Project type.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.VolumeProject("/home/admin/work/supervisely/projects/volumes", sly.OpenMode.READ)
            print(project.type)
            # Output: 'volumes'
        """
        return ProjectType.VOLUMES.value

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_volumes: Optional[bool] = True,
        log_progress: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        **kwargs,
    ) -> None:
        """
        Download volume project from Supervisely to the given directory.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Dataset IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_volumes: Download volume data files or not.
        :type download_volumes: :class:`bool`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: bool
        :param progress_cb: Function for tracking the download progress.
        :type progress_cb: tqdm or callable, optional

        :return: None
        :rtype: NoneType
        :Usage example:
        .. code-block:: python

                import supervisely as sly

                # Local destination Volume Project folder
                save_directory = "/home/admin/work/supervisely/source/vlm_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.VolumeProject.download(api, project_id, save_directory)
                project_fs = sly.VolumeProject(save_directory, sly.OpenMode.READ)
        """
        download_volume_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_volumes=download_volumes,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def download_bin(
        api: Api,
        project_id: int,
        dest_dir: Optional[str] = None,
        dataset_ids: Optional[List[int]] = None,
        download_volumes: bool = True,
        log_progress: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        return_bytesio: bool = False,
        schema_version: str = DEFAULT_VOLUME_SCHEMA_VERSION,
        *args,
        **kwargs,
    ) -> Union[str, io.BytesIO]:
        """
        Download a Volume Project snapshot into a Parquet-backed binary blob (`.tar.zst` file or in-memory BytesIO).

        The snapshot stores:

        - Project info and meta
        - Dataset tree (dataset infos)
        - Volume infos (optionally)
        - Volume annotations (for the included volumes)

        The resulting binary snapshot can be restored later with :func:`upload_bin`.

        :param api: Supervisely API client.
        :type api: :class:`~supervisely.api.api.Api`
        :param project_id: Source Volume Project ID on the server.
        :type project_id: int
        :param dest_dir: Local folder where the snapshot file will be written. Required when `return_bytesio=False`.
        :type dest_dir: str, optional
        :param dataset_ids: Optional list of dataset IDs to include. If provided, only these datasets will be included (recursively, preserving tree structure where applicable).
        :type dataset_ids: List[int], optional
        :param download_volumes: If False, only project/meta/dataset tree is stored (volume infos and annotations are skipped). This is useful for “structure-only” snapshots.
        :type download_volumes: bool, optional
        :param log_progress: If True, show a progress bar (unless a custom ``progress_cb`` is provided).
        :type log_progress: bool
        :param progress_cb: Optional callback (or tqdm-like object) called with incremental progress.
        :type progress_cb: tqdm or callable, optional
        :param return_bytesio: If True, return an in-memory :class:`io.BytesIO` with snapshot bytes. If False, write snapshot to ``dest_dir`` and return the file path.
        :type return_bytesio: bool, optional
        :param schema_version: Snapshot schema version. Controls the internal Parquet layout/fields. Supported values are the keys from :func:`~supervisely.project.volume_schema.get_volume_snapshot_schema` (currently: ``"v2.0.0"``).
        :type schema_version: str, optional
        :return: Snapshot file path (when ``return_bytesio=False``) or a BytesIO (when ``return_bytesio=True``).
        :rtype: str or io.BytesIO
        :raises ValueError: If ``dest_dir`` is not provided and ``return_bytesio`` is False.
        :raises RuntimeError: If required optional dependencies (e.g. pyarrow) are missing.

        :Usage example:

        .. code-block:: python

            import supervisely as sly
            import os

            api = sly.Api(os.environ["SERVER_ADDRESS"], os.environ["API_TOKEN"])

            # 1) Save snapshot to disk
            out_path = sly.VolumeProject.download_bin(
                api,
                project_id=123,
                dest_dir="/tmp/vol_project_snapshot",
                download_volumes=True,
                log_progress=True,
            )

            # 2) Create an in-memory snapshot (BytesIO) and restore it
            blob = sly.VolumeProject.download_bin(
                api,
                project_id=123,
                return_bytesio=True,
                download_volumes=False,  # structure-only
            )
            restored = sly.VolumeProject.upload_bin(api, blob, workspace_id=45, project_name="Restored")
        """

        pa = VolumeProject._require_pyarrow()
        snapshot_schema = get_volume_snapshot_schema(schema_version)

        if dest_dir is None and not return_bytesio:
            raise ValueError(
                "Local save directory dest_dir must be specified if return_bytesio is False"
            )

        ds_filters = (
            [{"field": "id", "operator": "in", "value": dataset_ids}]
            if dataset_ids is not None
            else None
        )

        project_info = api.project.get_info_by_id(project_id)
        project_meta = api.project.get_meta(project_id, with_settings=True)
        project_meta_obj = ProjectMeta.from_json(project_meta)
        dataset_infos = api.dataset.get_list(project_id, filters=ds_filters, recursive=True, include_custom_data=True)

        dataset_records = [dataset_info._asdict() for dataset_info in dataset_infos]
        volume_records: List[Dict] = []
        annotations: Dict[str, Dict] = {}
        key_id_map = KeyIdMap()

        for dataset_info in dataset_infos:
            if dataset_ids is not None and dataset_info.id not in dataset_ids:
                continue

            volumes = api.volume.get_list(dataset_info.id)
            if len(volumes) == 0:
                continue

            if not download_volumes:
                continue

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Collecting volumes from: {!r}".format(dataset_info.name),
                    total=len(volumes),
                )

            volume_ids = [volume_info.id for volume_info in volumes]
            ann_jsons = api.volume.annotation.download_bulk(dataset_info.id, volume_ids)

            # insert custom_data into ann_jsons (api does not return it in download_bulk atm)
            # Build mappings:
            # - volume_id -> ann_json
            # - volume_id -> {figure_id -> spatial_figure_dict}
            ann_by_volume_id: Dict[int, Dict[str, Any]] = {}
            spatial_figures_by_volume: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for ann_json in ann_jsons:
                volume_id = ann_json.get(ApiField.VOLUME_ID)
                if volume_id is None:
                    continue
                ann_by_volume_id[volume_id] = ann_json
                figures_list = ann_json.get(volume_constants.SPATIAL_FIGURES, []) or []
                fig_id_to_spatial_figure: Dict[int, Dict[str, Any]] = {}
                for spatial_figure in figures_list:
                    fig_id = spatial_figure.get("id")
                    if fig_id is not None:
                        fig_id_to_spatial_figure[fig_id] = spatial_figure
                spatial_figures_by_volume[volume_id] = fig_id_to_spatial_figure

            figures_dict = api.volume.figure.download(dataset_info.id, volume_ids)
            for volume_id, figure_infos in figures_dict.items():
                ann_json = ann_by_volume_id.get(volume_id)
                if ann_json is None:
                    continue
                fig_id_to_spatial_figure = spatial_figures_by_volume.get(volume_id, {})
                for figure_info in figure_infos:
                    spatial_figure = fig_id_to_spatial_figure.get(figure_info.id)
                    if spatial_figure is not None:
                        spatial_figure[ApiField.CUSTOM_DATA] = figure_info.custom_data

            for volume_info, ann_json in zip(volumes, ann_jsons):
                ann_dict = snapshot_schema.annotation_dict_from_raw(
                    api=api,
                    raw_ann_json=ann_json,
                    project_meta_obj=project_meta_obj,
                    key_id_map=key_id_map,
                )
                volume_records.append(volume_info._asdict())
                annotations[str(volume_info.id)] = ann_dict
                if progress_cb is not None:
                    progress_cb(1)
                if ds_progress is not None:
                    ds_progress(1)

        project_info_dict = project_info._asdict()
        project_info_dict[VersionSchemaField.SCHEMA_VERSION] = schema_version
        payload = {
            "project_info": project_info_dict,
            "project_meta": project_meta,
            "dataset_infos": dataset_records,
            "volume_infos": volume_records,
            "annotations": annotations,
        }
        blob = VolumeProject._serialize_payload_to_parquet_blob(pa, payload, snapshot_schema)

        if return_bytesio:
            stream = io.BytesIO(blob)
            stream.seek(0)
            return stream

        os.makedirs(dest_dir, exist_ok=True)
        file_path = os.path.join(dest_dir, f"{project_info.id}_{project_info.name}.arrow")
        with open(file_path, "wb") as out:
            out.write(blob)
        return file_path

    @staticmethod
    def upload(
        directory: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Tuple[int, str]:
        """
        Uploads volume project to Supervisely from the given directory.

        :param directory: Path to project directory.
        :type directory: :class:`str`
        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param workspace_id: Workspace ID, where project will be uploaded.
        :type workspace_id: :class:`int`
        :param project_name: Name of the project in Supervisely. Can be changed if project with the same name is already exists.
        :type project_name: :class:`str`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking the download progress.
        :type progress_cb: tqdm or callable, optional

        :return: Project ID and name. It is recommended to check that returned project name coincides with provided project name.
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Volume Project
            project_directory = "/home/admin/work/supervisely/source/vlm_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Volume Project
            project_id, project_name = sly.VolumeProject.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Volume Project"
            )
        """
        return upload_volume_project(
            dir=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload_bin(
        api: Api,
        file: Union[str, io.BytesIO],
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        skip_missed_entities: bool = False,
        *args,
        **kwargs,
    ) -> ProjectInfo:
        """
        Restore a volume project from a Parquet blob produced by :func:`download_bin`.

        :param api: Supervisely API client.
        :type api: :class:`~supervisely.api.api.Api`
        :param file: Snapshot file path (``.tar.zst``) or an in-memory :class:`io.BytesIO` stream.
        :type file: Union[str, io.BytesIO]
        :param workspace_id: Target workspace ID where the project will be created.
        :type workspace_id: int
        :param project_name: Optional new project name. If not provided, the name from the snapshot will be used. If the name already exists in the workspace, a free name will be chosen.
        :type project_name: str, optional
        :param log_progress: If True, show a progress bar (unless a custom ``progress_cb`` is provided).
        :type log_progress: bool
        :param progress_cb: Optional callback (or tqdm-like object) called with incremental progress.
        :type progress_cb: tqdm or callable, optional
        :param skip_missed_entities: If True, skip volumes that cannot be restored because their source hash is missing in the snapshot payload. If False, such cases raise an error.
        :type skip_missed_entities: bool
        :return: Info of the newly created project.
        :rtype: :class:`~supervisely.api.project_api.ProjectInfo`
        :raises RuntimeError: If the snapshot contains volumes without hashes and ``skip_missed_entities`` is False.
        """

        pa = VolumeProject._require_pyarrow()

        if isinstance(file, io.BytesIO):
            raw_data = file.getbuffer()
        else:
            with open(file, "rb") as src:
                raw_data = src.read()

        payload = VolumeProject._deserialize_payload_from_parquet(pa, raw_data)

        project_meta = ProjectMeta.from_json(payload["project_meta"])
        project_info: Dict = payload.get("project_info", {})
        dataset_records: List[Dict] = payload.get("dataset_infos", [])
        volume_records: List[Dict] = payload.get("volume_infos", [])
        annotations: Dict[str, Dict] = payload.get("annotations", {})

        project_title = project_name or project_info.get("name")
        if api.project.exists(workspace_id, project_title):
            project_title = api.project.get_free_name(workspace_id, project_title)
        src_project_desc = project_info.get("description")
        new_project_info = api.project.create(
            workspace_id,
            project_title,
            ProjectType.VOLUMES,
            description=src_project_desc,
            readme=project_info.get("readme"),
        )
        api.project.update_meta(new_project_info.id, project_meta)

        custom_data = new_project_info.custom_data
        source_project_id = payload["project_info"].get("id")
        version_info = payload["project_info"].get("version") or {}
        custom_data["restored_from"] = {
            "project_id": source_project_id,
            "version_num": version_info.get("version"),
        }
        original_custom_data = payload["project_info"].get("custom_data") or {}
        custom_data.update(original_custom_data)
        api.project.update_custom_data(new_project_info.id, custom_data, silent=True)

        dataset_mapping: Dict[int, sly.DatasetInfo] = {}
        sorted_datasets = sorted(
            dataset_records,
            key=lambda data: (data.get("parent_id") is not None, data.get("parent_id") or 0),
        )
        for dataset_data in sorted_datasets:
            parent_ds_info = dataset_mapping.get(dataset_data.get("parent_id"))
            new_parent_id = parent_ds_info.id if parent_ds_info else None
            new_dataset_info = api.dataset.create(
                project_id=new_project_info.id,
                name=dataset_data.get("name"),
                description=dataset_data.get("description"),
                parent_id=new_parent_id,
                custom_data=dataset_data.get("custom_data"),
            )
            dataset_mapping[dataset_data.get("id")] = new_dataset_info

        volume_mapping: Dict[int, VolumeInfo] = {}
        volumes_by_dataset: Dict[int, List[Dict]] = defaultdict(list)
        for volume_data in volume_records:
            volumes_by_dataset[volume_data.get("dataset_id")].append(volume_data)

        for old_dataset_id, dataset_volumes in volumes_by_dataset.items():
            new_dataset_info = dataset_mapping.get(old_dataset_id)
            if new_dataset_info is None:
                continue

            dataset_volumes_to_upload: List[Dict] = []
            missing_names: List[str] = []
            for vol in dataset_volumes:
                if vol.get("hash"):
                    dataset_volumes_to_upload.append(vol)
                else:
                    missing_names.append(vol.get("name") or str(vol.get("id")))

            if missing_names:
                if skip_missed_entities:
                    for vol_name in missing_names:
                        logger.warning(
                            "Volume %r skipped during restoration because its source hash is unavailable.",
                            vol_name,
                        )
                    if len(dataset_volumes_to_upload) == 0:
                        continue
                else:
                    raise RuntimeError(
                        "Cannot restore volumes without available hash. Missing volume names: {}".format(
                            ", ".join(missing_names)
                        )
                    )

            hashes = [volume.get("hash") for volume in dataset_volumes_to_upload]
            names = [volume.get("name") for volume in dataset_volumes_to_upload]
            metas = [volume.get("meta") for volume in dataset_volumes_to_upload]

            ds_progress = progress_cb
            if log_progress and progress_cb is None:
                ds_progress = tqdm_sly(
                    desc="Uploading volumes to {!r}".format(new_dataset_info.name),
                    total=len(dataset_volumes_to_upload),
                )

            new_volume_infos = api.volume.upload_hashes(
                new_dataset_info.id,
                names=names,
                hashes=hashes,
                metas=metas,
                progress_cb=ds_progress,
            )

            for old_volume, new_volume in zip(dataset_volumes_to_upload, new_volume_infos):
                volume_mapping[old_volume.get("id")] = new_volume

        for volume_id_str, ann_json in annotations.items():
            new_volume_info = volume_mapping.get(int(volume_id_str))
            if new_volume_info is None:
                if skip_missed_entities:
                    logger.warning(
                        "Annotation for volume %s skipped because the source volume was not restored.",
                        volume_id_str,
                    )
                continue
            ann_json["volumeId"] = new_volume_info.id
            ann = VolumeAnnotation.from_json(ann_json, project_meta, None)
            api.volume.annotation.append(new_volume_info.id, ann, None)

        return api.project.get_info_by_id(new_project_info.id)

    @staticmethod
    def _require_pyarrow():
        try:
            import pyarrow as pa  # pylint: disable=import-error
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VolumeProject binary versioning requires the optional dependency 'pyarrow'. "
                "Install it with `pip install pyarrow` to use download_bin/upload_bin."
            ) from exc
        return pa

    @staticmethod
    def _serialize_payload_to_parquet_blob(pa_module, payload: Dict[str, Dict], snapshot_schema) -> bytes:
        dataset_records: List[Dict] = payload.get("dataset_infos", []) or []
        volume_records: List[Dict] = payload.get("volume_infos", []) or []
        annotations_dict: Dict[str, Dict] = payload.get("annotations", {}) or {}

        dataset_rows = [snapshot_schema.dataset_row_from_record(r) for r in dataset_records]
        dataset_table = pa_module.Table.from_pylist(
            dataset_rows, schema=snapshot_schema.datasets_table_schema(pa_module)
        )

        volume_rows = [snapshot_schema.volume_row_from_record(r) for r in volume_records]
        volume_table = pa_module.Table.from_pylist(
            volume_rows, schema=snapshot_schema.volumes_table_schema(pa_module)
        )

        ann_rows = []
        for volume_id_str, ann in annotations_dict.items():
            try:
                src_volume_id = int(volume_id_str)
            except (TypeError, ValueError):
                continue
            ann_rows.append(
                snapshot_schema.annotation_row_from_dict(src_volume_id=src_volume_id, annotation=ann)
            )
        annotations_table = pa_module.Table.from_pylist(
            ann_rows, schema=snapshot_schema.annotations_table_schema(pa_module)
        )

        sections = [
            (
                VolumeProject._SECTION_PROJECT_INFO,
                VolumeProject._json_bytes(payload.get("project_info", {})),
            ),
            (
                VolumeProject._SECTION_PROJECT_META,
                VolumeProject._json_bytes(payload.get("project_meta", {})),
            ),
            (
                VolumeProject._SECTION_DATASETS,
                VolumeProject._table_to_parquet_bytes(pa_module, dataset_table),
            ),
            (
                VolumeProject._SECTION_VOLUMES,
                VolumeProject._table_to_parquet_bytes(pa_module, volume_table),
            ),
            (
                VolumeProject._SECTION_ANNOTATIONS,
                VolumeProject._table_to_parquet_bytes(pa_module, annotations_table),
            ),
        ]

        return VolumeProject._assemble_sections(sections)

    @staticmethod
    def _build_table(pa_module, columns: Dict[str, Tuple[List, Any]]):
        arrays = {}
        for name, (values, dtype) in columns.items():
            arrays[name] = pa_module.array(values, type=dtype)
        return pa_module.table(arrays)

    @staticmethod
    def _table_to_parquet_bytes(pa_module, table) -> bytes:
        from pyarrow import parquet as pq  # pylint: disable=import-error

        sink = pa_module.BufferOutputStream()
        pq.write_table(table, sink)
        return sink.getvalue().to_pybytes()

    @staticmethod
    def _parquet_bytes_to_table(pa_module, data: bytes):
        if not data:
            return pa_module.table({})
        from pyarrow import parquet as pq  # pylint: disable=import-error

        buffer = pa_module.BufferReader(data)
        return pq.read_table(buffer)

    @staticmethod
    def _json_dumps(data) -> str:
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _json_bytes(data) -> bytes:
        return VolumeProject._json_dumps(data).encode("utf-8")

    @staticmethod
    def _assemble_sections(sections: List[Tuple[int, bytes]]) -> bytes:
        if len(sections) > 255:
            raise RuntimeError("Too many sections for VolumeProject binary payload")
        buffer = io.BytesIO()
        buffer.write(VolumeProject._SERIALIZATION_MAGIC)
        buffer.write(struct.pack(">B", VolumeProject._SERIALIZATION_VERSION))
        buffer.write(struct.pack(">B", len(sections)))
        for section_type, payload in sections:
            if payload is None:
                payload = b""
            buffer.write(struct.pack(">B", section_type))
            buffer.write(struct.pack(">Q", len(payload)))
            buffer.write(payload)
        return buffer.getvalue()

    @staticmethod
    def _parse_parquet_sections(raw_data) -> Dict[int, bytes]:
        magic = VolumeProject._SERIALIZATION_MAGIC
        view = raw_data if isinstance(raw_data, memoryview) else memoryview(raw_data)
        header_len = len(magic) + 2
        if len(view) < header_len:
            logger.warning(
                f"VolumeProject binary payload too small: {len(view)} bytes (need >= {header_len}). First bytes(hex)={view[: min(len(view), 16)].tobytes().hex()}",
            )
            raise RuntimeError("Corrupted VolumeProject binary payload")
        if view[: len(magic)].tobytes() != magic:
            found = view[: len(magic)].tobytes()
            logger.warning(
                f"VolumeProject binary payload magic mismatch. expected={magic.hex()} found={found.hex()} total_bytes={len(view)} prefix16(hex)={view[:16].tobytes().hex()}",
            )
            raise RuntimeError(
                "Unsupported VolumeProject binary payload format (magic mismatch). "
                "Expected magic={!r}, found={!r}".format(magic, found)
            )

        offset = len(magic)
        version = view[offset]
        offset += 1
        if version != VolumeProject._SERIALIZATION_VERSION:
            logger.warning(
                "VolumeProject binary payload version mismatch. expected=%d found=%d total_bytes=%d",
                VolumeProject._SERIALIZATION_VERSION,
                version,
                len(view),
            )
            raise RuntimeError(
                "Unsupported VolumeProject binary payload version: {}".format(version)
            )

        section_count = view[offset]
        offset += 1
        sections: Dict[int, bytes] = {}
        for _ in range(section_count):
            if offset + 9 > len(view):
                raise RuntimeError("Corrupted VolumeProject binary payload")
            section_type = view[offset]
            offset += 1
            length = int.from_bytes(view[offset : offset + 8], "big")
            offset += 8
            if offset + length > len(view):
                raise RuntimeError("Corrupted VolumeProject binary payload")
            sections[section_type] = view[offset : offset + length].tobytes()
            offset += length
        return sections

    @staticmethod
    def _deserialize_payload_from_parquet(pa_module, raw_data) -> Dict:
        sections = VolumeProject._parse_parquet_sections(raw_data)

        try:
            project_info = json.loads(sections[VolumeProject._SECTION_PROJECT_INFO].decode("utf-8"))
            project_meta = json.loads(sections[VolumeProject._SECTION_PROJECT_META].decode("utf-8"))
        except KeyError as exc:
            raise RuntimeError("VolumeProject payload missing metadata section") from exc

        if VolumeProject._SECTION_DATASETS not in sections:
            logger.warning("VolumeProject blob has no datasets section; treating as empty.")
        if VolumeProject._SECTION_VOLUMES not in sections:
            logger.warning("VolumeProject blob has no volumes section; treating as empty.")
        if VolumeProject._SECTION_ANNOTATIONS not in sections:
            logger.warning("VolumeProject blob has no annotations section; treating as empty.")

        dataset_table = VolumeProject._parquet_bytes_to_table(
            pa_module, sections.get(VolumeProject._SECTION_DATASETS, b"")
        )
        volume_table = VolumeProject._parquet_bytes_to_table(
            pa_module, sections.get(VolumeProject._SECTION_VOLUMES, b"")
        )
        annotations_table = VolumeProject._parquet_bytes_to_table(
            pa_module, sections.get(VolumeProject._SECTION_ANNOTATIONS, b"")
        )

        dataset_records: List[Dict] = []
        if dataset_table is not None and dataset_table.num_rows:
            col_json = (
                VersionSchemaField.JSON
                if VersionSchemaField.JSON in dataset_table.column_names
                else "json"
            )
            if col_json in dataset_table.column_names:
                dataset_jsons = dataset_table.column(col_json).to_pylist()
                dataset_records = [json.loads(item) for item in dataset_jsons]

        volume_records: List[Dict] = []
        if volume_table is not None and volume_table.num_rows:
            col_json = (
                VersionSchemaField.JSON
                if VersionSchemaField.JSON in volume_table.column_names
                else "json"
            )
            if col_json in volume_table.column_names:
                volume_jsons = volume_table.column(col_json).to_pylist()
                volume_records = [json.loads(item) for item in volume_jsons]

        annotations: Dict[str, Dict] = {}
        if annotations_table is not None and annotations_table.num_rows:
            col_vol_id = (
                VersionSchemaField.SRC_VOLUME_ID
                if VersionSchemaField.SRC_VOLUME_ID in annotations_table.column_names
                else "volume_id"
            )
            col_ann = (
                VersionSchemaField.ANNOTATION
                if VersionSchemaField.ANNOTATION in annotations_table.column_names
                else "annotation"
            )
            if col_vol_id in annotations_table.column_names and col_ann in annotations_table.column_names:
                annotation_ids = annotations_table.column(col_vol_id).to_pylist()
                annotation_payloads = annotations_table.column(col_ann).to_pylist()
                for volume_id, annotation_json in zip(annotation_ids, annotation_payloads):
                    if volume_id is None:
                        continue
                    annotations[str(volume_id)] = json.loads(annotation_json)

        return {
            "project_info": project_info,
            "project_meta": project_meta,
            "dataset_infos": dataset_records,
            "volume_infos": volume_records,
            "annotations": annotations,
        }

    @staticmethod
    def _load_mask_geometries(api: Api, ann: VolumeAnnotation, key_id_map: KeyIdMap) -> None:
        for sf in ann.spatial_figures:
            if sf.geometry.name() != Mask3D.name():
                continue
            api.volume.figure.load_sf_geometry(sf, key_id_map)

    @staticmethod
    def get_train_val_splits_by_count(project_dir: str, train_count: int, val_count: int) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            "Static method 'get_train_val_splits_by_count()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            "Static method 'get_train_val_splits_by_tag()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_collections(
        project_dir: str,
        train_collections: List[int],
        val_collections: List[int],
        project_id: int,
        api: Api,
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_collections()' is not supported for VolumeProject class now."
        )

    @staticmethod
    async def download_async(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'download_async()' is not supported for VolumeProject class now."
        )


def download_volume_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_volumes: Optional[bool] = True,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Download volume project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_volumes: Include volumes in the download.
    :type download_volumes: bool, optional
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
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        dest_dir = 'your/local/dest/dir'

        # Download volume project
        project_id = 18532
        project_info = api.project.get_info_by_id(project_id)
        num_volumes = project_info.items_count

        p = tqdm(desc="Downloading volume project", total=num_volumes)
        sly.download_volume_project(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """

    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VolumeProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    datasets_infos = []
    if dataset_ids is not None:
        for ds_id in dataset_ids:
            datasets_infos.append(api.dataset.get_info_by_id(ds_id))
    else:
        datasets_infos = api.dataset.get_list(project_id)

    for dataset in datasets_infos:
        dataset_fs: VolumeDataset = project_fs.create_dataset(dataset.name)
        volumes = api.volume.get_list(dataset.id)

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Downloading volumes from: {!r}".format(dataset.name),
                total=len(volumes),
            )
        for batch in batched(volumes, batch_size=LOG_BATCH_SIZE):
            volume_ids = [volume_info.id for volume_info in batch]
            volume_names = [volume_info.name for volume_info in batch]

            ann_jsons = api.volume.annotation.download_bulk(dataset.id, volume_ids)

            for volume_id, volume_name, volume_info, ann_json in zip(
                volume_ids, volume_names, batch, ann_jsons
            ):
                if volume_name != ann_json[ApiField.VOLUME_NAME]:
                    raise RuntimeError(
                        "Error in api.volume.annotation.download_batch: broken order"
                    )
                try:
                    ann = VolumeAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "volume_id": volume_id,
                            "volume_name": volume_name,
                            "ann_json": ann_json,
                        },
                    )
                    raise e

                volume_file_path = dataset_fs.generate_item_path(volume_name)
                if download_volumes is True:
                    header = None
                    item_progress = None
                    if ds_progress is not None:
                        item_progress = tqdm_sly(
                            desc=f"Downloading '{volume_name}'",
                            total=volume_info.sizeb,
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        )
                        api.video.download_path(volume_id, volume_file_path, item_progress)
                    else:
                        api.volume.download_path(volume_id, volume_file_path)
                else:
                    touch(volume_file_path)
                    header = _create_volume_header(ann)

                mask_ids = []
                mask_paths = []
                mesh_ids = []
                mesh_paths = []
                for sf in ann.spatial_figures:
                    figure_id = key_id_map.get_figure_id(sf.key())
                    if sf.geometry.name() == Mask3D.name():
                        mask_ids.append(figure_id)
                        figure_path = dataset_fs.get_mask_path(volume_name, sf)
                        mask_paths.append(figure_path)
                    if sf.geometry.name() == ClosedSurfaceMesh.name():
                        mesh_ids.append(figure_id)
                        figure_path = dataset_fs.get_interpolation_path(volume_name, sf)
                        mesh_paths.append(figure_path)
                
                figs = api.volume.figure.download(dataset.id, [volume_id], skip_geometry=True)
                figs = figs.get(volume_id, {})
                figs_ids_map = {fig.id: fig for fig in figs}
                for ann_fig in ann.figures + ann.spatial_figures:
                    fig = figs_ids_map.get(ann_fig.geometry.sly_id)
                    ann_fig.custom_data.update(fig.custom_data)

                api.volume.figure.download_stl_meshes(mesh_ids, mesh_paths)
                api.volume.figure.download_sf_geometries(mask_ids, mask_paths)

                # prepare a list of paths where converted STLs will be stored
                nrrd_paths = []
                for file in mesh_paths:
                    file = re.sub(r"\.[^.]+$", ".nrrd", file)
                    file = change_directory_at_index(file, "mask", -3)  # change destination folder
                    nrrd_paths.append(file)

                stl_converter.to_nrrd(mesh_paths, nrrd_paths, header=header)

                ann, meta = api.volume.annotation._update_on_transfer(
                    "download", ann, project_fs.meta, nrrd_paths
                )

                project_fs.set_meta(meta)

                dataset_fs.add_item_file(
                    volume_name,
                    volume_file_path,
                    ann=ann,
                    _validate_item=False,
                )

                if progress_cb is not None:
                    progress_cb(1)

            if log_progress:
                ds_progress(len(batch))

    project_fs.set_key_id_map(key_id_map)


def load_figure_data(
    api: Api, volume_file_path: str, spatial_figure: VolumeFigure, key_id_map: KeyIdMap
):
    """
    Load data into figure geometry.

    :param api: Supervisely API address and token.
    :type api: Api
    :param volume_file_path: Path to Volume file location
    :type volume_file_path: str
    :param spatial_figure: Spatial figure
    :type spatial_figure: VolumeFigure object
    :param key_id_map: Mapped keys and IDs
    :type key_id_map: KeyIdMap object
    """
    figure_id = key_id_map.get_figure_id(spatial_figure.key())
    figure_path = "{}_mask3d/".format(volume_file_path[:-5]) + f"{figure_id}.nrrd"
    api.volume.figure.download_stl_meshes([figure_id], [figure_path])
    Mask3D.from_file(spatial_figure, figure_path)


# TODO: add methods to convert to 3d masks


def upload_volume_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    project_fs = VolumeProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VOLUMES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    if progress_cb is not None:
        log_progress = False

    item_id_dct, anns_paths_dct, interpolation_dirs_dct, mask_dirs_dct = {}, {}, {}, {}

    for dataset_fs in project_fs.datasets:
        dataset_fs: VolumeDataset
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths, mask_dirs, interpolation_dirs = [], [], [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(img_path)
            ann_paths.append(ann_path)
            interpolation_dirs.append(dataset_fs.get_interpolation_dir(item_name))
            mask_dirs.append(dataset_fs.get_mask_dir(item_name))

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm_sly(
                desc="Uploading volumes to {!r}".format(dataset.name),
                total=len(item_paths),
                position=0,
            )

        item_infos = api.volume.upload_nrrd_series_paths(
            dataset.id, names, item_paths, ds_progress, log_progress
        )
        volume_ids = [item_info.id for item_info in item_infos]

        anns_progress = None
        if log_progress is True or progress_cb is not None:
            anns_progress = tqdm_sly(
                desc="Uploading annotations to {!r}".format(dataset.name),
                total=len(volume_ids),
                leave=False,
            )
        api.volume.annotation.upload_paths(
            volume_ids,
            ann_paths,
            project_fs.meta,
            interpolation_dirs,
            anns_progress,
            mask_dirs,
        )

    return project.id, project.name


def _create_volume_header(ann: VolumeAnnotation) -> Dict:
    """
    Create volume header to use in STL converter when downloading project without volumes.

    :param ann: VolumeAnnotation object
    :type ann: VolumeAnnotation
    :return: header with Volume meta parameters
    :rtype: Dict
    """
    header = {}
    header["sizes"] = numpy.array([value for _, value in ann.volume_meta["dimensionsIJK"].items()])
    world_matrix = ann.volume_meta["IJK2WorldMatrix"]
    header["space directions"] = numpy.array(
        [world_matrix[i : i + 3] for i in range(0, len(world_matrix) - 4, 4)]
    )
    header["space origin"] = numpy.array(
        [world_matrix[i + 3] for i in range(0, len(world_matrix) - 4, 4)]
    )
    if ann.volume_meta["ACS"] == "RAS":
        header["space"] = "right-anterior-superior"
    elif ann.volume_meta["ACS"] == "LAS":
        header["space"] = "left-anterior-superior"
    elif ann.volume_meta["ACS"] == "LPS":
        header["space"] = "left-posterior-superior"
    return header
