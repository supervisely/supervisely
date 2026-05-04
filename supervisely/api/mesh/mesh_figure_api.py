# coding: utf-8
from __future__ import annotations

from typing import Dict, List, Optional

from requests_toolbelt import MultipartEncoder

from supervisely._utils import batched
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.api.module_api import ApiField
from supervisely.mesh_annotation.mesh_figure import MeshFigure
from supervisely.mesh_annotation.mesh_indices import decode_mesh_indices, encode_mesh_indices
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshFigureApi(FigureApi):
    """API for mesh annotation figures."""

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
        figures: List[MeshFigure],
        key_id_map: KeyIdMap,
    ) -> None:
        regular_figures_json = []
        regular_keys = []
        mesh_keys = []
        mesh_indices = []
        mesh_figures_json = []

        for figure in figures:
            figure_json = figure.to_json(key_id_map)
            if figure_json.get(ApiField.GEOMETRY_TYPE) == "mesh":
                mesh_keys.append(figure.key())
                mesh_indices.append(figure.geometry.indices)
                figure_json.pop(ApiField.GEOMETRY, None)
                mesh_figures_json.append(figure_json)
            else:
                regular_keys.append(figure.key())
                regular_figures_json.append(figure_json)

        self._append_bulk(mesh_id, regular_figures_json, regular_keys, key_id_map)
        self._append_bulk(mesh_id, mesh_figures_json, mesh_keys, key_id_map)

        figure_ids = [key_id_map.get_figure_id(key) for key in mesh_keys]
        if len(figure_ids) != 0:
            self.upload_indices_batch(figure_ids, mesh_indices)

    def download_indices_batch(self, figure_ids: List[int]) -> List[List[int]]:
        """Download mesh figure index geometry as raw little-endian uint32 data."""
        geometries = {}
        for figure_id, part in self._download_geometries_generator(figure_ids):
            geometries[figure_id] = decode_mesh_indices(part.content)

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

        for batch_ids, batch_indices in zip(
            batched(figure_ids, batch_size=100),
            batched(indices_batch, batch_size=100),
        ):
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
