# coding: utf-8

import asyncio
import os
import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.api.module_api import ApiField
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import (
    change_directory_at_index,
    dir_exists,
    file_exists,
    get_file_name,
    list_files,
    silent_remove,
)
from supervisely.io.json import load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume import stl_converter
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)


class VolumeAnnotationAPI(EntityAnnotationAPI):
    """
    :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>` for a single volume. :class:`VolumeAnnotationAPI<VolumeAnnotationAPI>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        volume_id = 19581134
        ann_info = api.volume.annotation.download(volume_id)
    """

    _method_download_bulk = "volumes.annotations.bulk.info"
    _entity_ids_str = ApiField.VOLUME_IDS

    def download(self, volume_id: int):
        """
        Download information about VolumeAnnotation by volume ID from API.
        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :return: Information about VolumeAnnotation in json format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from pprint import pprint

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            ann_info = api.volume.annotation.download(volume_id)
            print(ann_info)
            # Output:
            # {
            #     'createdAt': '2023-03-29T12:30:37.078Z',
            #     'datasetId': 61803,
            #     'description': '',
            #     'objects': [],
            #     'planes': [],
            #     'spatialFigures': [],
            #     'tags': [{'createdAt': '2023-04-03T13:21:53.368Z',
            #             'id': 12259702,
            #             'labelerLogin': 'almaz',
            #             'name': 'info',
            #             'tagId': 385328,
            #             'updatedAt': '2023-04-03T13:21:53.368Z',
            #             'value': 'age 31'}],
            #     'updatedAt': '2023-03-29T12:30:37.078Z',
            #     'volumeId': 19581134,
            #     'volumeMeta': {
            #             'ACS': 'RAS',
            #             'IJK2WorldMatrix': [0.7617, 0, 0,
            #                                 -194.2384, 0, 0.76171,
            #                                 0, -217.5384, 0,
            #                                 0, 2.5, -347.75,
            #                                 0, 0, 0, 1],
            #             'channelsCount': 1,
            #             'dimensionsIJK': {'x': 512, 'y': 512, 'z': 139},
            #             'intensity': {'max': 3071, 'min': -3024},
            #             'rescaleIntercept': 0,
            #             'rescaleSlope': 1,
            #             'windowCenter': 23.5,
            #             'windowWidth': 6095
            # },
            #     'volumeName': 'CTChest.nrrd'
            # }
        """

        volume_info = self._api.volume.get_info_by_id(volume_id)
        return self._download(volume_info.dataset_id, volume_id)

    def append(
        self, volume_id: int, ann: VolumeAnnotation, key_id_map: KeyIdMap = None, volume_info=None
    ):
        """
        Loads VolumeAnnotation to a given volume ID in the API.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param ann: VolumeAnnotation object.
        :type ann: VolumeAnnotation
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            api.volume.annotation.append(volume_id, volume_ann)
        """
        if ann.spatial_figures:
            figures = ann.figures + ann.spatial_figures
        else:
            figures = ann.figures

        if volume_info is None:
            volume_info = self._api.volume.get_info_by_id(volume_id)
        self._append(
            self._api.volume.tag,
            self._api.volume.object,
            self._api.volume.figure,
            volume_info.project_id,
            volume_info.dataset_id,
            volume_id,
            ann.tags,
            ann.objects,
            figures,
            key_id_map,
        )

    def upload_paths(
        self,
        volume_ids: List[int],
        ann_paths: List[str],
        project_meta: ProjectMeta,
        interpolation_dirs: Optional[List[str]] = None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        mask_dirs: Optional[List[str]] = None,
    ) -> None:
        """
        Loads VolumeAnnotations from a given paths to a given volumes IDs in the API. Volumes IDs must be from one dataset.

        :param volume_ids: Volumes IDs in Supervisely.
        :type volume_ids: List[int]
        :param ann_paths: Paths to annotation files
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VolumeAnnotations
        :type project_meta: ProjectMeta
        :param interpolation_dirs: Paths to dirs with interpolation STL files
        :type interpolation_dirs: List[str], optional
        :param progress_cb: Function for tracking download progress
        :type progress_cb: tqdm or callable, optional
        :param mask_dirs: Paths to dirs with Mask3D geometries
        :type mask_dirs: List[str], optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_ids = [121236918, 121236919]
            ann_pathes = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            api.volume.annotation.upload_paths(volume_ids, ann_pathes, meta)
        """

        # use in updating project metadata
        project_id = self._api.volume.get_info_by_id(volume_ids[0]).project_id

        if interpolation_dirs is None:
            interpolation_dirs = [None] * len(ann_paths)

        if mask_dirs is None:
            mask_dirs = [None] * len(ann_paths)

        key_id_map = KeyIdMap()
        for volume_id, ann_path, interpolation_dir, mask_dir in zip(
            volume_ids, ann_paths, interpolation_dirs, mask_dirs
        ):
            ann_json = load_json_file(ann_path)
            ann = VolumeAnnotation.from_json(ann_json, project_meta)

            geometries_dict = {}
            stl_paths = []
            nrrd_paths = []
            keep_nrrd_paths = []

            if interpolation_dir is not None and dir_exists(interpolation_dir):
                (
                    stl_paths,
                    nrrd_paths,
                    keep_nrrd_paths,
                ) = self._prepare_convertation_paths(interpolation_dir, ann)

                if len(stl_paths) != 0:
                    stl_converter.to_nrrd(stl_paths, nrrd_paths)
                    ann, project_meta = self._update_on_transfer(
                        "upload", ann, project_meta, nrrd_paths, project_id
                    )

            # list all Mask3D geometries
            if mask_dir is not None and dir_exists(mask_dir):
                mask_paths = list_files(mask_dir, valid_extensions=[".nrrd"])
                geometries_dict.update(Mask3D._bytes_from_nrrd_batch(mask_paths))
                # it is not recommended to change the original composition of the project directory files
                # for this purpose delete the files created during conversion
                for nrrd_path in nrrd_paths:
                    if nrrd_path not in keep_nrrd_paths:
                        silent_remove(nrrd_path)

            # add geometries into spatial figure objects
            for sf in ann.spatial_figures:
                try:
                    geometry_bytes = geometries_dict[sf.key().hex]
                    mask3d = Mask3D.from_bytes(geometry_bytes)
                    sf._set_3d_geometry(mask3d)
                except (KeyError, TypeError) as e:
                    if isinstance(e, TypeError):
                        logger.warning(
                            f"Skipping spatial figure for class '{sf.volume_object.obj_class.name}': {str(e)}"
                        )
                    # skip figures that doesn't need to update geometry
                    # for example for old geometries that are stored in JSON (KeyError)
                    continue

            self.append(volume_id, ann, key_id_map)

            if progress_cb is not None:
                progress_cb(1)

    def _update_on_transfer(
        self,
        transfer_type: Literal["download", "upload"],
        ann: VolumeAnnotation,
        project_meta: ProjectMeta,
        nrrd_paths: List[str],
        project_id: Optional[int] = None,
    ) -> Tuple[VolumeAnnotation, ProjectMeta]:
        """
        Create new ObjClass and VolumeFigure annotations for converted STL.
        Replace ClosedMeshSurface spatial figures with Mask3D.
        Update the ann, project_meta, and key_id_map.

        :param transfer_type: Defines the process during which the update will be performed ("download" or "upload").
        :type transfer_type: Literal["download", "upload"]
        :param ann: The VolumeAnnotation object to update.
        :type ann: VolumeAnnotation
        :param project_meta: The ProjectMeta object.
        :type project_meta: ProjectMeta
        :param nrrd_paths: Paths to the converted NRRD files from STL.
        :type nrrd_paths: List[str]
        :param project_id: The Project ID to update metadata on upload (optional).
        :type project_id: int, optional
        :return: A tuple containing the updated ann and project_meta objects.
        :rtype: Tuple[VolumeAnnotation, ProjectMeta]
        """

        for nrrd_path in nrrd_paths:
            object_key = None
            custom_data = None

            # searching connection between interpolation and spatial figure in annotations and set its object_key
            for sf in ann.spatial_figures:
                if sf.key().hex == get_file_name(nrrd_path):
                    object_key = sf.parent_object.key()
                    custom_data = sf.custom_data
                    break

            if object_key:
                for obj in ann.objects:
                    if obj.key() == object_key:
                        class_title = obj.obj_class.name
                        break

            else:
                raise Exception(
                    f"Can't find volume object for Mask3D from unknown file '{nrrd_path}'. Please check the project structure."
                )

            class_created = False
            new_obj_class = project_meta.get_obj_class(f"{class_title}_mask_3d")

            if new_obj_class is None:
                new_obj_class = ObjClass(f"{class_title}_mask_3d", Mask3D)
                project_meta = project_meta.add_obj_class(new_obj_class)
                class_created = True

            geometry = Mask3D(np.zeros((3, 3, 3), dtype=np.bool_))

            if transfer_type == "download":
                new_object = VolumeObject(new_obj_class, mask_3d=geometry, custom_data=custom_data)
            elif transfer_type == "upload":
                if class_created:
                    self._api.project.update_meta(project_id, project_meta)
                new_object = VolumeObject(new_obj_class)
                new_object.figure = VolumeFigure(new_object, geometry, key=sf.key(), custom_data=custom_data)

            # add new Volume object to VolumeAnnotation with spatial figure
            ann = ann.add_objects([new_object])

            if transfer_type == "download":
                # as a sf.key() changes, we need to rename both files
                os.rename(
                    nrrd_path, f"{os.path.dirname(nrrd_path)}/{new_object.figure.key().hex}.nrrd"
                )
                stl_path = re.sub(r"\.[^.]+$", ".stl", nrrd_path)
                stl_path = change_directory_at_index(stl_path, "interpolation", -3)
                os.rename(
                    stl_path, f"{os.path.dirname(stl_path)}/{new_object.figure.key().hex}.stl"
                )
            # remove STL spatial figure from VolumeAnnotation
            if sf:
                ann.spatial_figures.remove(sf)

        return ann, project_meta

    def append_objects(
        self,
        volume_id: int,
        objects: Union[List[VolumeObject], VolumeObjectCollection],
        key_id_map: Optional[KeyIdMap] = None,
    ) -> None:
        """
        Add new VolumeObjects to a volume annotation in Supervisely project.

        :param volume_id: The ID of the volume.
        :type volume_id: int
        :param objects: New volume objects.
        :type objects: List[VolumeObject] or VolumeObjectCollection
        :param key_id_map: The KeyIdMap (optional).
        :type key_id_map: KeyIdMap, optional
        :return: None
        :rtype: NoneType

        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            if sly.is_development():
               load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            volume_id = 151344
            volume_info = api.volume.get_info_by_id(volume_id)
            mask_3d_path = "data/mask/lung.nrrd"
            lung_obj_class = sly.ObjClass("lung", sly.Mask3D)
            lung = sly.VolumeObject(lung_obj_class, mask_3d=mask_3d_path)
            objects = sly.VolumeObjectCollection([lung])
            api.volume.annotation.append_objects(volume_info.id, objects)
        """

        sf_figures = []
        for volume_object in objects:
            if volume_object.obj_class.geometry_type in (Mask3D, AnyGeometry):
                if isinstance(volume_object.figure.geometry, Mask3D):
                    sf_figures.append(volume_object.figure)

        volume_meta = self._api.volume.get_info_by_id(volume_id).meta
        ann = VolumeAnnotation(volume_meta, objects, spatial_figures=sf_figures)
        self.append(volume_id, ann, key_id_map)

    @staticmethod
    def _prepare_convertation_paths(
        interpolation_dir: str, ann: VolumeAnnotation
    ) -> Tuple[List, List, List]:
        """
        Check dir and create paths for NRRD files if STL files need to be converted.

        :param interpolation_dir: Path to dir with interpolation STL files
        :type interpolation_dir: str
        :param ann: VolumeAnnotation object
        :type ann: VolumeAnnotation
        :return: Paths to STL and NRRD files used in the conversion process
        :rtype: Tuple[List, List, List]
        """
        stl_paths_in = list_files(interpolation_dir, valid_extensions=[".stl"])
        nrrd_paths = []
        stl_paths = []
        keep_nrrd_paths = []  # to keep original composition of the project
        for stl_path in stl_paths_in:
            # check if this is really STL file
            with open(stl_path, "rb") as file:
                header = file.read(84)
                if b"solid" in header:
                    stl_paths.append(stl_path)
                else:
                    continue
            nrrd_path = re.sub(r"\.[^.]+$", ".nrrd", stl_path)
            nrrd_path = change_directory_at_index(nrrd_path, "mask", -3)
            nrrd_paths.append(nrrd_path)
            if file_exists(nrrd_path):
                keep_nrrd_paths.append(nrrd_path)
                # to prevent duplication if STL is already converted to NRRD on export
                for sf in ann.spatial_figures:
                    if sf.key().hex == get_file_name(nrrd_path) and isinstance(sf.geometry, Mask3D):
                        stl_paths.remove(stl_path)
                        nrrd_paths.remove(nrrd_path)
                        keep_nrrd_paths.remove(nrrd_path)
        return stl_paths, nrrd_paths, keep_nrrd_paths

    async def download_async(
        self,
        volume_id: int,
        semaphore: Optional[asyncio.Semaphore] = None,
        integer_coords: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Dict:
        """
        Download information about VolumeAnnotation by volume ID from API asynchronously.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param semaphore: Semaphore to limit the number of parallel downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param integer_coords: If True, returns coordinates as integers for objects. If False, returns as floats.
        :type integer_coords: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about VolumeAnnotation in json format
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 198702499
            loop = sly.utils.get_or_create_event_loop()
            ann_info = loop.run_until_complete(api.volume.annotation.download_async(volume_id))
        """
        return await self.download_bulk_async(
            volume_ids=[volume_id],
            semaphore=semaphore,
            integer_coords=integer_coords,
            progress_cb=progress_cb,
        )

    async def download_bulk_async(
        self,
        volume_ids: List[int],
        semaphore: Optional[asyncio.Semaphore] = None,
        integer_coords: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> List[Dict]:
        """
        Download information about VolumeAnnotation in bulk by volume IDs from API asynchronously.

        :param volume_ids: List of Volume IDs in Supervisely. All volumes must be from the same dataset.
        :type volume_ids: int
        :param semaphore: Semaphore to limit the number of parallel downloads.
        :type semaphore: asyncio.Semaphore, optional
        :param integer_coords: If True, returns coordinates as integers for objects. If False, returns as floats.
        :type integer_coords: bool, optional
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :return: Information about VolumeAnnotations in json format
        :rtype: :class:`list`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_ids = [198702499, 198702500, 198702501]
            loop = sly.utils.get_or_create_event_loop()
            ann_infos = loop.run_until_complete(api.volume.annotation.download_bulk_async(volume_ids))
        """
        if semaphore is None:
            semaphore = self._api.get_default_semaphore()

        async def fetch_with_semaphore(batch):
            async with semaphore:
                json_data = {
                    self._entity_ids_str: batch,
                    ApiField.INTEGER_COORDS: integer_coords,
                }
                response = await self._api.post_async(
                    self._method_download_bulk,
                    json=json_data,
                )
                if progress_cb is not None:
                    progress_cb(len(batch))
                return response.json()

        tasks = [fetch_with_semaphore(batch) for batch in batched(volume_ids)]
        responses = await asyncio.gather(*tasks)
        json_response = [item for response in responses for item in response]
        return json_response
