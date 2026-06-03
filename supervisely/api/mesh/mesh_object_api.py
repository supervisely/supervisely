# coding: utf-8
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

from tqdm import tqdm

from requests_toolbelt import MultipartEncoder

from supervisely._utils import batched
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.module_api import ApiField
from supervisely.mesh_annotation.mesh_indices import (
    decode_mesh_indices,
    encode_mesh_indices,
)
from supervisely.task.progress import update_progress
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshObjectApi(FigureApi):
    """API for mesh annotation objects.

    Following the image annotation model, a mesh "object" is a single per-label
    entity that references its mesh entity directly (``entityId`` + ``classId``),
    exactly like an image figure. The labeling UI calls these *objects*, hence the
    name (see the ``# @TODO: rename to object like in labeling UI`` note on
    :class:`~supervisely.api.image_api.ImageApi`). Mesh index geometry is stored as
    a separate blob in geometry storage, analogous to alpha-mask geometry.
    """

    def create(
        self,
        mesh_id: int,
        geometry_json: Dict,
        geometry_type: str,
        class_id: int,
        custom_data: Optional[dict] = None,
    ) -> int:
        figure_json = {
            ApiField.ENTITY_ID: mesh_id,
            ApiField.GEOMETRY_TYPE: geometry_type,
            ApiField.GEOMETRY: geometry_json,
        }
        if class_id is not None:
            figure_json[ApiField.CLASS_ID] = class_id
        if custom_data is not None:
            figure_json[ApiField.CUSTOM_DATA] = custom_data
        return self.create_bulk([figure_json], entity_id=mesh_id)[0]

    def append_bulk(
        self,
        mesh_id: int,
        figures_json: List[Dict],
        figures_keys: List,
        key_id_map: KeyIdMap,
    ) -> None:
        """Create mesh objects (image-style figures) and map label keys to figure IDs."""
        self._append_bulk(mesh_id, figures_json, figures_keys, key_id_map)

    def download_indices_batch(
        self,
        figure_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[List[int]]:
        """Download mesh object index geometry as raw little-endian uint32 data.

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
        """Upload mesh object index geometry as raw little-endian uint32 data."""
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
