# coding: utf-8
import re
from typing import List, Union, Optional
from uuid import UUID
from requests_toolbelt import MultipartDecoder, MultipartEncoder
from supervisely.io.fs import ensure_base_path, get_nested_dicts_data
from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.volume_annotation.plane import Plane
from supervisely.geometry.mask_3d import Mask3D
import supervisely.volume_annotation.constants as constants
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume.nrrd_encoder import encode


class VolumeFigureApi(FigureApi):
    """
    :class:`VolumeFigure<supervisely.volume_annotation.volume_figure.VolumeFigure>` for a single volume.
    """

    def create(
        self,
        volume_id: int,
        object_id: int,
        plane_name: str,
        slice_index: int,
        geometry_json: dict,
        geometry_type,
        # track_id=None,
    ):
        """
        Create new VolumeFigure for given slice in given volume ID.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param object_id: ID of the object to which the VolumeFigure belongs.
        :type object_id: int
        :param plane_name: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>` of the slice in volume.
        :type plane_name: str
        :param slice_index: Number of the slice to add VolumeFigure.
        :type slice_index: int
        :param geometry_json: Parameters of geometry for VolumeFigure.
        :type geometry_json: dict
        :param geometry_type: Type of VolumeFigure geometry.
        :type geometry_type: str
        :return: New figure ID
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.volume_annotation.plane import Plane

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            object_id = 5565016
            slice_index = 0
            plane_name = Plane.AXIAL
            geometry_json = {'points': {'exterior': [[500, 500], [1555, 1500]], 'interior': []}}
            geometry_type = 'rectangle'

            figure_id = api.volume.figure.create(
                volume_id,
                object_id,
                plane_name,
                slice_index,
                geometry_json,
                geometry_type
            ) # 87821207
        """

        Plane.validate_name(plane_name)

        return super().create(
            volume_id,
            object_id,
            {
                constants.SLICE_INDEX: slice_index,
                constants.NORMAL: Plane.get_normal(plane_name),
                # for backward compatibility
                ApiField.META: {
                    constants.SLICE_INDEX: slice_index,
                    constants.NORMAL: Plane.get_normal(plane_name),
                },
            },
            geometry_json,
            geometry_type,
            # track_id,
        )

    def append_bulk(
        self, volume_id: int, figures: List[VolumeFigure], key_id_map: KeyIdMap
    ):
        """
        Add VolumeFigures to given Volume by ID.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int        
        :param figures: List of VolumeFigure objects.
        :type figures: List[VolumeFigure]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import numpy as np

            from supervisely.volume_annotation.plane import Plane

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 19370
            volume_id = 19617444

            key_id_map = sly.KeyIdMap()

            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)
            volume_obj_collection = vol_ann.objects.to_json()
            vol_obj = sly.VolumeObject.from_json(volume_obj_collection[1], project_meta)

            geometry = sly.Mask3D(np.zeros(3, 3, 3). dtype=np.bool_)  
            
            figure = sly.VolumeFigure(
                vol_obj,
                geometry,
                None,
                None,
            )

            api.volume.figure.append_bulk(volume_id, [figure], key_id_map)
        """

        if not figures:
            return

        keys = []
        mask3d_keys = []
        figures_json = []
        mask3d_figures_json = []

        for figure in figures:
            if figure.geometry.name() == Mask3D.name():
                mask3d_keys.append(figure.key())
                mask3d_figures_json.append(figure.to_json(key_id_map, save_meta=True))
            else:
                keys.append(figure.key())
                figures_json.append(figure.to_json(key_id_map, save_meta=True))
        # Figure is missing required field \"meta.normal\"","index":0}}
        self._append_bulk(volume_id, figures_json, keys, key_id_map)
        if mask3d_figures_json:
            self._append_bulk_mask3d(
                volume_id, mask3d_figures_json, mask3d_keys, key_id_map
            )

    def _download_geometries_batch(self, ids: List[int]):
        """
        Private method. Download figures geometries with given IDs from storage.
        """

        for batch_ids in batched(ids):
            response = self._api.post(
                "figures.bulk.download.geometry", {ApiField.IDS: batch_ids}
            )
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                figure_id = int(
                    re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1]
                )
                yield figure_id, part

    def download_sf_geometries(self, ids: List[int], paths: Optional[List[str]] = None):
        """
        Download spatial figures geometry for the specified figure IDs. 
        Saves them to the specified paths if paths are passed.

        :param ids: VolumeFigure ID in Supervisely
        :type ids: int
        :param paths: List of paths to save
        :type paths: List[str]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            STORAGE_DIR = sly.app.get_data_dir()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)            
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            ids = []
            paths = []
            
            for sp_figure in vol_ann.spatial_figures:
                figure_id = key_id_map.get_figure_id(sp_figure.key())
                ids.append(figure_id)
                paths.appen(f"{STORAGE_DIR}/{figure_id}.nrrd")                        
            api.volume.figure.download_sf_geometries(ids, paths)
        """

        if not ids:
            return

        id_to_data = {}
        
        if paths:
            if len(ids) != len(paths):
                raise RuntimeError(
                    'Can not match "ids" and "paths" lists, len(ids) != len(paths)'
                )
            id_to_path = {id: path for id, path in zip(ids, paths)}
            
        for figure_id, resp_part in self._download_geometries_batch(ids):
            id_to_data[figure_id] = resp_part.content
            if paths:
                ensure_base_path(id_to_path[figure_id])
                with open(id_to_path[figure_id], "wb") as w:
                    w.write(resp_part.content)
        
        return id_to_data

    def _append_bulk_mask3d(
        self,
        entity_id: int,
        figures_json: List,
        figures_keys: List,
        key_id_map: KeyIdMap,
        field_name=ApiField.ENTITY_ID,
    ):
        """
        The same method as _append_bulk but for spatial figures. Uploads figures with geometry to given Volume by ID.

        :param entity_id: Volume ID.
        :type entity_id: int
        :param figures_json: List of figure dicts.
        :type figures_json: list
        :param figures_keys: List of figure keys as UUID.
        :type figures_keys: list
        :param key_id_map: KeyIdMap object (dict with bidict values)
        :type key_id_map: KeyIdMap
        :param field_name: field name for request body
        :type field_name: str
        :rtype: :class:`NoneType`
        :Usage example:
        """
        figures_count = len(figures_json)
        if figures_count == 0:
            return

        empty_figures = []
        for figure in figures_json:
            empty_figures.append(
                {
                    "objectId": figure["objectId"],
                    "geometryType": Mask3D.name(),
                    "tool": Mask3D.name(),
                    "entityId": entity_id,
                }
            )
        for batch_keys, batch_jsons in zip(
            batched(figures_keys, batch_size=100),
            batched(empty_figures, batch_size=100),
        ):
            resp = self._api.post(
                "figures.bulk.add",
                {field_name: entity_id, ApiField.FIGURES: batch_jsons},
            )
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)

                for figure_json in figures_json:
                    if figure_json.get("key") == key.hex:
                        geometry = get_nested_dicts_data(
                            figure_json, "geometry", "mask_3d", "data"
                        )
                        geometry = Mask3D.base64_2_data(geometry)
                        geometry_bytes = encode(geometry)

                        self.upload_sf_geometries([key], [geometry_bytes], key_id_map)

    def upload_sf_geometries(
        self,
        spatial_figures: List[Union[VolumeFigure, str]],
        geometries: List[bytes],
        key_id_map: KeyIdMap,
    ):
        """
        Upload spatial figures geometry as bytes to storage by given ID.

        :param spatial_figures: List with VolumeFigure objects or figure key
        :type spatial_figures: List[Union[VolumeFigure, str]]
        :param geometries: List with geometries, which represented as NRRD in byte format.
        :type geometries: List[bytes]
        :param key_id_map: KeyIdMap object (dict with bidict values)
        :type key_id_map: KeyIdMap
        :rtype: :class:`NoneType`
        :Usage example:
        """
        if len(spatial_figures) == 0:
            return

        for sf, geometry_bytes in zip(spatial_figures, geometries):
            if type(sf) == UUID:
                figure_id = key_id_map.get_figure_id(sf)
            else:
                figure_id = key_id_map.get_figure_id(sf.key())
            content_dict = {
                ApiField.FIGURE_ID: str(figure_id),
                ApiField.GEOMETRY: (str(figure_id), geometry_bytes, "application/sla"),
            }
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post("figures.bulk.upload.geometry", encoder)
