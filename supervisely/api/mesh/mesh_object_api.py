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


class MeshObjectApi(FigureApi):
    """API for mesh annotation objects.

    A mesh annotation is a flat list of objects, each referencing its mesh entity
    directly (``entityId`` + ``classId``). The nested "figures" entity used by
    video/pointcloud annotations is not applicable to mesh annotations. Object
    index geometry is stored as a separate blob in geometry storage.
    """

    def create(
        self,
        mesh_id: int,
        geometry_json: Dict,
        geometry_type: str,
        class_id: int,
        custom_data: Optional[dict] = None,
    ) -> int:
        """
        Create a single mesh object and return its newly assigned ID.

        :param mesh_id: ID of the mesh entity the object belongs to.
        :type mesh_id: int
        :param geometry_json: Object geometry in JSON format.
        :type geometry_json: dict
        :param geometry_type: Geometry type identifier.
        :type geometry_type: str
        :param class_id: ID of the object class.
        :type class_id: int
        :param custom_data: Arbitrary custom data attached to the object.
        :type custom_data: dict, optional
        :returns: ID of the created mesh object.
        :rtype: int
        """
        object_json = {
            ApiField.ENTITY_ID: mesh_id,
            ApiField.GEOMETRY_TYPE: geometry_type,
            ApiField.GEOMETRY: geometry_json,
        }
        if class_id is not None:
            object_json[ApiField.CLASS_ID] = class_id
        if custom_data is not None:
            object_json[ApiField.CUSTOM_DATA] = custom_data
        return self.create_bulk([object_json], entity_id=mesh_id)[0]

    def append_bulk(self, mesh_id: int, objects_json: List[Dict]) -> List[int]:
        """Create mesh objects and return their assigned IDs, ordered like ``objects_json``."""
        if len(objects_json) == 0:
            return []
        return self.create_bulk(objects_json, entity_id=mesh_id)

    def download_indices_batch(
        self,
        object_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[List[int]]:
        """Download mesh object index geometry as raw little-endian uint32 data.

        Progress is updated by one for each downloaded object geometry.
        """
        geometries = {}
        for object_id, part in self._download_geometries_generator(object_ids):
            geometries[object_id] = decode_mesh_indices(part.content)
            if progress_cb is not None:
                update_progress(progress_cb, 1)

        if len(geometries) != len(object_ids):
            raise RuntimeError("Not all mesh geometries were downloaded")
        return [geometries[object_id] for object_id in object_ids]

    def upload_indices_batch(self, object_ids: List[int], indices_batch: List[List[int]]) -> None:
        """Upload mesh object index geometry as raw little-endian uint32 data."""
        if len(object_ids) != len(indices_batch):
            raise ValueError(
                f"object_ids and indices_batch must have the same length: "
                f"{len(object_ids)} != {len(indices_batch)}."
            )

        for batch in batched(list(zip(object_ids, indices_batch)), batch_size=100):
            batch_ids, batch_indices = zip(*batch)
            fields = []
            for object_id, indices in zip(batch_ids, batch_indices):
                fields.append((ApiField.FIGURE_ID, str(object_id)))
                fields.append(
                    (
                        ApiField.GEOMETRY,
                        (str(object_id), encode_mesh_indices(indices), "application/octet-stream"),
                    )
                )
            encoder = MultipartEncoder(fields=fields)
            self._api.post("figures.bulk.upload.geometry", encoder)
