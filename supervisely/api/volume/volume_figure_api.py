# coding: utf-8
import re
import os
import tempfile
from uuid import UUID
from numpy import ndarray, uint8
from typing import List, Dict, Union
from requests_toolbelt import MultipartDecoder, MultipartEncoder

import supervisely.volume_annotation.constants as constants
from supervisely.io.fs import ensure_base_path, file_exists, list_files, get_file_name
from supervisely._utils import batched
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.api.entity_annotation.figure_api import FigureApi
from supervisely.volume_annotation.plane import Plane
from supervisely.geometry.mask_3d import Mask3D
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
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

    def append_bulk(self, volume_id: int, figures: List[VolumeFigure], key_id_map: KeyIdMap):
        """
        Add VolumeFigures to given Volume by ID.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :param figures: List of VolumeFigure objects.
        :type figures: List[VolumeFigure]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

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

            figure = sly.VolumeFigure(
                vol_obj,
                sly.Rectangle(20, 20, 129, 200),
                sly.Plane.AXIAL,
                45,
            )

            api.volume.figure.append_bulk(volume_id, [figure], key_id_map)
        """

        if not figures:
            return

        keys = []
        mask3d_keys = []
        figures_json = []
        mask3d_figures = []

        for figure in figures:
            if figure.geometry.name() == Mask3D.name():
                mask3d_keys.append(figure.key())
                mask3d_figures.append(figure)
            else:
                keys.append(figure.key())
                figures_json.append(figure.to_json(key_id_map, save_meta=True))
        # Figure is missing required field \"meta.normal\"","index":0}}
        self._append_bulk(volume_id, figures_json, keys, key_id_map)
        if mask3d_figures:
            self._append_bulk_mask3d(volume_id, mask3d_figures, mask3d_keys, key_id_map)

    def _download_geometries_batch(self, ids: List[int]):
        """
        Private method. Download figures geometries with given IDs from storage.
        """

        for batch_ids in batched(ids):
            response = self._api.post("figures.bulk.download.geometry", {ApiField.IDS: batch_ids})
            decoder = MultipartDecoder.from_response(response)
            for part in decoder.parts:
                content_utf8 = part.headers[b"Content-Disposition"].decode("utf-8")
                # Find name="1245" preceded by a whitespace, semicolon or beginning of line.
                # The regex has 2 capture group: one for the prefix and one for the actual name value.
                figure_id = int(re.findall(r'(^|[\s;])name="(\d*)"', content_utf8)[0][1])
                yield figure_id, part

    def download_stl_meshes(self, ids: List[int], paths: List[str]):
        """
        Download STL meshes for the specified figure IDs and saves them to the specified paths.

        :param ids: VolumeFigure ID in Supervisely.
        :type ids: int
        :param paths: List of paths to download.
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
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            for sp_figure in vol_ann.spatial_figures:
                figure_id = key_id_map.get_figure_id(sp_figure.key())
                id_to_paths[figure_id] = f"{STORAGE_DIR}/{sp_figure.key().hex}.stl"
            if id_to_paths:
                api.volume.figure.download_stl_meshes(*zip(*id_to_paths.items()))
        """

        if len(ids) == 0:
            return
        if len(ids) != len(paths):
            raise RuntimeError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')

        id_to_path = {id: path for id, path in zip(ids, paths)}
        for img_id, resp_part in self._download_geometries_batch(ids):
            ensure_base_path(id_to_path[img_id])
            with open(id_to_path[img_id], "wb") as w:
                w.write(resp_part.content)

    def interpolate(self, volume_id: int, spatial_figure: VolumeFigure, key_id_map: KeyIdMap):
        """
        Interpolate a spatial figure with a ClosedSurfaceMesh geometry.

        :param volume_id: VolumeFigure ID in Supervisely.
        :type volume_id: int
        :param spatial_figure: Spatial figure to interpolate.
        :type spatial_figure: VolumeFigure
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            for sp_figure in vol_ann.spatial_figures:
                res = volume_figure_api.interpolate(volume_id, sp_figure, key_id_map)
        """

        if type(spatial_figure._geometry) != ClosedSurfaceMesh:
            raise TypeError(
                "Interpolation can be created only for figures with geometry ClosedSurfaceMesh"
            )
        object_id = key_id_map.get_object_id(spatial_figure.volume_object.key())
        response = self._api.post(
            "figures.volumetric_interpolation",
            {ApiField.VOLUME_ID: volume_id, ApiField.OBJECT_ID: object_id},
        )
        return response.content

    def _upload_meshes_batch(self, figure2bytes):
        """
        Private method. Upload figures geometry by given ID to storage.

        :param figure2bytes: Dictionary with figures IDs and geometries.
        :type figure2bytes: dict
        :rtype: :class:`NoneType`
        :Usage example:
        """

        for figure_id, figure_bytes in figure2bytes.items():
            content_dict = {
                ApiField.FIGURE_ID: str(figure_id),
                ApiField.GEOMETRY: (str(figure_id), figure_bytes, "application/sla"),
            }
            encoder = MultipartEncoder(fields=content_dict)
            resp = self._api.post("figures.bulk.upload.geometry", encoder)

    def upload_stl_meshes(
        self,
        volume_id: int,
        spatial_figures: List[VolumeFigure],
        key_id_map: KeyIdMap,
        interpolation_dir=None,
    ):
        """
        Upload existing interpolations or create on the fly and and add them to empty mesh figures.

        :param volume_id: VolumeFigure ID in Supervisely.
        :type volume_id: int
        :param spatial_figures: List of spatial figures to upload.
        :type spatial_figures: List[VolumeFigure]
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19371414
            project_id = 17215

            volume = api.volume.get_info_by_id(volume_id)

            key_id_map = sly.KeyIdMap()
            project_meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(project_meta_json)

            vol_ann_json = api.volume.annotation.download(volume_id)
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)
            sp_figures = vol_ann.spatial_figures

            res = volume_figure_api.upload_stl_meshes(volume_id, sp_figures, key_id_map)
        """

        if len(spatial_figures) == 0:
            return

        figure2bytes = {}
        for sp in spatial_figures:
            figure_id = key_id_map.get_figure_id(sp.key())
            if interpolation_dir is not None:
                meth_path = os.path.join(interpolation_dir, sp.key().hex + ".stl")
                if file_exists(meth_path):
                    with open(meth_path, "rb") as in_file:
                        meth_bytes = in_file.read()
                    figure2bytes[figure_id] = meth_bytes
            # else - no stl file
            if figure_id not in figure2bytes:
                meth_bytes = self.interpolate(volume_id, sp, key_id_map)
                figure2bytes[figure_id] = meth_bytes
        self._upload_meshes_batch(figure2bytes)

    def _append_bulk_mask3d(
        self,
        entity_id: int,
        figures: List,
        figures_keys: List,
        key_id_map: KeyIdMap,
        field_name=ApiField.ENTITY_ID,
    ):
        """The same method as _append_bulk but for spatial figures. Uploads figures to given Volume by ID.
        You need to upload the geometry right after figures will be created

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
        figures_count = len(figures)
        if figures_count == 0:
            return

        empty_figures = []
        for figure in figures:
            empty_figures.append(
                {
                    "objectId": figure.volume_object.to_json(key_id_map)["id"],
                    "geometryType": Mask3D.name(),
                    "tool": Mask3D.name(),
                    "entityId": entity_id,
                }
            )
        for batch_keys, batch_jsons in zip(
            batched(figures_keys, batch_size=100), batched(empty_figures, batch_size=100)
        ):
            resp = self._api.post(
                "figures.bulk.add",
                {field_name: entity_id, ApiField.FIGURES: batch_jsons},
            )
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)

                for figure in figures:
                    if figure.key() == key:
                        geometry = figure.geometry.data
                        geometry_bytes = encode(geometry.astype(uint8))

                        self.upload_sf_geometries([key], {key: geometry_bytes}, key_id_map)

    def upload_sf_geometries(
        self,
        spatial_figures: List[UUID],
        geometries: Dict[UUID, bytes],
        key_id_map: KeyIdMap,
    ):
        """
        Upload geometries into spatial figures in project as bytes using their keys.

        :param spatial_figures: List with figure UUID keys
        :type spatial_figures: List[UUID]
        :param geometries: Dict where keys are UUID of spatial figure, and values are geometries, which represented as NRRD in bytes.
        :type geometries: Dict[UUID, bytes]
        :param key_id_map: KeyIdMap object (dict with bidict values)
        :type key_id_map: KeyIdMap
        :rtype: :class:`NoneType`
        :Usage example:
        .. code-block:: python

            import numpy as np
            import supervisely as sly
            from supervisely.volume.nrrd_encoder import encode

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 23772225
            project_id = 28159
            geometries = {}
            key_id_map = sly.KeyIdMap()
            geometry_bytes = encode(np.random.randint(2, size=(20, 20, 20), dtype=np.uint8))

            vol_ann_json = api.volume.annotation.download(volume_id)
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)
            spatial_figures = [sp_figure.key() for sp_figure in ann.spatial_figures]
            for figure in spatial_figures:
                geometries[figure] = geometry_bytes
            api.volume.figure.upload_sf_geometries(spatial_figures, geometries, key_id_map)
        """

        if not spatial_figures:
            return

        for sf in spatial_figures:
            figure_id = key_id_map.get_figure_id(sf)
            geometry_bytes = geometries.get(sf)
            content_dict = {
                ApiField.FIGURE_ID: str(figure_id),
                ApiField.GEOMETRY: (str(figure_id), geometry_bytes, "application/sla"),
            }
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post("figures.bulk.upload.geometry", encoder)

    def download_sf_geometries(self, ids: List[int], paths: List[str]):
        """
        Download spatial figures geometry for the specified figure IDs and saves them to the specified paths

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
            id_to_paths = {}
            vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, project_meta, key_id_map)

            for sp_figure in vol_ann.spatial_figures:
                figure_id = key_id_map.get_figure_id(sp_figure.key())
                id_to_paths[figure_id] = f"{STORAGE_DIR}/{sp_figure.key().hex}.stl"
            if id_to_paths:
                api.volume.figure.download_sf_geometries(*zip(*id_to_paths.items()))
        """

        if not ids:
            return
        if len(ids) != len(paths):
            raise RuntimeError('Can not match "ids" and "paths" lists, len(ids) != len(paths)')

        id_to_path = {id: path for id, path in zip(ids, paths)}
        for figure_id, resp_part in self._download_geometries_batch(ids):
            ensure_base_path(id_to_path[figure_id])
            with open(id_to_path[figure_id], "wb") as w:
                w.write(resp_part.content)

    def read_sf_geometries(self, path: str) -> Dict[str, bytes]:
        """
        Read geometries as bytes in dictionary and maps them to figure UUID hex value.
        NRRD file must be named with UUID hex value.

        :param path: Path to file or dir with files
        :type path: str
        :return: Dictionary with geometries
        :rtype: Dict[str, bytes]
        """
        geometries_dict = {}
        if os.path.isdir(path):
            files_list = list_files(path)
        else:
            files_list = [path]
        for nrrd_file in files_list:
            key = get_file_name(nrrd_file)
            with open(nrrd_file, "rb") as file:
                geometry_bytes = file.read()
            geometries_dict[key] = geometry_bytes
        return geometries_dict

    def copy_geometry_to_figure(self, spatial_figure: VolumeFigure, key_id_map: KeyIdMap):
        """
        Download geometry by figure ID from existent figure in ecosystems project and load this data into new VolumeFigure object

        :param spatial_figure: Spatial figure object from VolumeAnnotation
        :type spatial_figure: VolumeFigure
        :param key_id_map: Mapped keys and IDs
        :type key_id_map: KeyIdMap object
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            figure_id = key_id_map.get_figure_id(spatial_figure.key())
            figure_path = f"{temp_dir}/{spatial_figure.key().hex}.nrrd"
            self.download_sf_geometries([figure_id], [figure_path])
            Mask3D.to_figure_from_file(spatial_figure, figure_path)

    def append_geometry_to_figure(
        self, spatial_figure: VolumeFigure, geometry: Union[str, ndarray, bytes]
    ):
        """
        Load geometry from file into VolumeFigure object

        :param spatial_figure: Spatial figure object from VolumeAnnotation
        :type spatial_figure: VolumeFigure
        :param geometry: Spatial figure object from VolumeAnnotation
        :type geometry: VolumeFigure
        """
        if isinstance(geometry, str):
            Mask3D.to_figure_from_file(spatial_figure, geometry)
        if isinstance(geometry, ndarray):
            Mask3D.to_figure_from_array(spatial_figure, geometry)
        if isinstance(geometry, bytes):
            Mask3D.to_figure_from_bytes(spatial_figure, geometry)
