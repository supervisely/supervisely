# coding: utf-8
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Union

from tqdm import tqdm

from requests_toolbelt import MultipartEncoder

from supervisely._utils import batched
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.module_api import ApiField
from supervisely.geometry.constants import GEOMETRY_SHAPE, GEOMETRY_TYPE
from supervisely.geometry.mesh import Mesh
from supervisely.mesh_annotation.mesh_indices import (
    decode_mesh_indices,
    encode_mesh_indices,
)
from supervisely.task.progress import update_progress
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshFigureApi(FigureApi):
    """Internal API for mesh annotation figure rows."""

    def create(
        self,
        mesh_id: int,
        object_id: int,
        geometry_json: Dict,
        geometry_type: str,
        track_id: Optional[int] = None,
        custom_data: Optional[dict] = None,
    ) -> int:
        return super().create(
            mesh_id,
            object_id,
            {},
            geometry_json,
            geometry_type,
            track_id,
            custom_data=custom_data,
        )

    def append_bulk(
        self,
        mesh_id: int,
        labels: Iterable,
        key_id_map: KeyIdMap,
    ) -> None:
        regular_figures_json = []
        regular_keys = []
        mesh_keys = []
        mesh_indices = []
        mesh_figures_json = []

        for label in labels:
            object_id = key_id_map.get_object_id(label.key())
            if object_id is None:
                raise RuntimeError(
                    "Can not upload mesh label figure: object ID not found for key {}".format(
                        label.key()
                    )
                )

            geometry_json = deepcopy(label.geometry.to_json())
            geometry_json.pop(GEOMETRY_TYPE, None)
            geometry_json.pop(GEOMETRY_SHAPE, None)
            figure_json = {
                ApiField.OBJECT_ID: object_id,
                ApiField.GEOMETRY_TYPE: label.geometry.geometry_name(),
                ApiField.GEOMETRY: geometry_json,
            }
            if label.priority is not None:
                figure_json[ApiField.PRIORITY] = label.priority
            if label.custom_data:
                figure_json[ApiField.CUSTOM_DATA] = label.custom_data

            if figure_json.get(ApiField.GEOMETRY_TYPE) == Mesh.geometry_name():
                mesh_keys.append(label.key())
                mesh_indices.append(label.geometry.indices)
                figure_json.pop(ApiField.GEOMETRY, None)
                mesh_figures_json.append(figure_json)
            else:
                regular_keys.append(label.key())
                regular_figures_json.append(figure_json)

        self._append_bulk(mesh_id, regular_figures_json, regular_keys, key_id_map)
        self._append_bulk(mesh_id, mesh_figures_json, mesh_keys, key_id_map)

        figure_ids = [key_id_map.get_figure_id(key) for key in mesh_keys]
        if len(figure_ids) != 0:
            self.upload_indices_batch(figure_ids, mesh_indices)

    def download_indices_batch(
        self,
        figure_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[List[int]]:
        """Download mesh figure index geometry as raw little-endian uint32 data.

        Progress is updated by one for each downloaded figure geometry.
        """
        geometries = {}
        for figure_id, part in self._download_geometries_generator(figure_ids):
            geometries[figure_id] = decode_mesh_indices(part.content)
            if progress_cb is not None:
                update_progress(progress_cb, 1)

        if len(geometries) != len(figure_ids):
            raise RuntimeError("Not all mesh geometries were downloaded")
        return [geometries[figure_id] for figure_id in figure_ids]

    def upload_indices_batch(self, figure_ids: List[int], indices_batch: List[List[int]]) -> None:
        """Upload mesh figure index geometry as raw little-endian uint32 data."""
        if len(figure_ids) != len(indices_batch):
            raise ValueError(
                f"figure_ids and indices_batch must have the same length: "
                f"{len(figure_ids)} != {len(indices_batch)}."
            )

        for batch in batched(list(zip(figure_ids, indices_batch)), batch_size=100):
            batch_ids, batch_indices = zip(*batch)
            fields = []
            for figure_id, indices in zip(batch_ids, batch_indices):
                fields.append((ApiField.FIGURE_ID, str(figure_id)))
                fields.append(
                    (
                        ApiField.GEOMETRY,
                        (str(figure_id), encode_mesh_indices(indices), "application/octet-stream"),
                    )
                )
            encoder = MultipartEncoder(fields=fields)
            self._api.post("figures.bulk.upload.geometry", encoder)
