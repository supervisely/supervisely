# coding: utf-8
from __future__ import annotations

from typing import Dict, List, Optional

from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.mesh_annotation.mesh_figure import MeshFigure
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
        keys = []
        figures_json = []
        for figure in figures:
            keys.append(figure.key())
            figures_json.append(figure.to_json(key_id_map))
        self._append_bulk(mesh_id, figures_json, keys, key_id_map)
