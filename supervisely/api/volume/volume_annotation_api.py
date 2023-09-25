# coding: utf-8


from typing import List, Optional, Union, Callable, Tuple, Literal

from tqdm import tqdm
import numpy as np
import os
import re


from supervisely.project.project_meta import ProjectMeta
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely.geometry.geometry import Geometry
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.mask_3d import Mask3D

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.io.json import load_json_file
from supervisely.io.fs import dir_exists, get_file_name, list_files, file_exists
from supervisely.volume import stl_converter

# from uuid import UUID


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
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

    def append(self, volume_id: int, ann: VolumeAnnotation, key_id_map: KeyIdMap = None):
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19581134
            api.volume.annotation.append(volume_id, volume_ann)
        """
        if ann.spatial_figures:
            figures = ann.figures + ann.spatial_figures
        else:
            figures = ann.figures

        info = self._api.volume.get_info_by_id(volume_id)
        self._append(
            self._api.volume.tag,
            self._api.volume.object,
            self._api.volume.figure,
            info.project_id,
            info.dataset_id,
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
        :param ann_paths: Paths to annotations on local machine
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VolumeAnnotations
        :type project_meta: ProjectMeta
        :param interpolation_dirs: Paths to interpolations on local machine
        :type interpolation_dirs: List[str], optional
        :param progress_cb: Function for tracking download progress
        :type progress_cb: tqdm or callable, optional
        :param mask_dirs: Paths to 3D Mask geometries on local machine
        :type mask_dirs: List[str], optional
        :return: None
        :rtype: :class:`NoneType`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_ids = [121236918, 121236919]
            ann_pathes = ['/home/admin/work/supervisely/example/ann1.json', '/home/admin/work/supervisely/example/ann2.json']
            api.volume.annotation.upload_paths(volume_ids, ann_pathes, meta)
        """

        # for use in updating project metadata, if necessary
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

            if interpolation_dir is not None and dir_exists(interpolation_dir):
                # list all STL files that need to be converted
                stl_paths = list_files(interpolation_dir)
                nrrd_paths = []
                # check if the STL + NRRD pair already exists
                for stl_path in stl_paths:
                    stl_path = re.sub(r"\.[^.]+$", ".stl", stl_path)
                    nrrd_path = re.sub(r"\.[^.]+$", ".nrrd", stl_path)
                    if file_exists(stl_path) and file_exists(nrrd_path):
                        stl_paths.remove(stl_path)
                        stl_paths.remove(nrrd_path)
                    else:
                        # if only STL exists - will be converted
                        nrrd_paths.append(nrrd_path)

                if len(stl_paths) != 0:
                    stl_converter.to_nrrd(stl_paths, nrrd_paths)
                    ann, project_meta = self._update_on_transfer(
                        "upload", ann, project_meta, nrrd_paths, key_id_map, project_id
                    )

                # list all Mask 3D geometries for interpolations that already been stored in NRRD
                stored_geometries = [
                    x for x in list_files(interpolation_dir) if x not in nrrd_paths
                ]
                geometries_dict.update(Geometry.bytes_from_file_batch(stored_geometries))

            # list all Mask 3D geometries
            if mask_dir is not None and dir_exists(mask_dir):
                mask_paths = list_files(mask_dir)
                geometries_dict.update(Geometry.bytes_from_file_batch(mask_paths))

            # add geometries into spatial figure objects
            for sf in ann.spatial_figures:
                try:
                    geometry_bytes = geometries_dict[sf.key().hex]
                except KeyError:  # skip figures that doesn't need to update geometry
                    continue
                maks3d = Mask3D.from_bytes(geometry_bytes)
                sf.set_geometry(maks3d)

            self.append(volume_id, ann, key_id_map)

            if progress_cb is not None:
                progress_cb(1)

    def _update_on_transfer(
        self,
        transfer_type: Literal["download", "upload"],
        ann: VolumeAnnotation,
        project_meta: ProjectMeta,
        nrrd_paths: List[str],
        key_id_map: KeyIdMap,
        project_id: Optional[int] = None,
    ) -> Tuple[VolumeAnnotation, ProjectMeta]:
        """
        Create new ObjClass and VolumeFigure annotations for converted STL.
        Replace ClosedMeshSurface spatial figures with Mask 3D.
        Update the ann, project_meta, and key_id_map.

        :param transfer_type: Defines the process during which the update will be performed ("download" or "upload").
        :type transfer_type: Literal["download", "upload"]
        :param ann: The VolumeAnnotation object to update.
        :type ann: VolumeAnnotation
        :param project_meta: The ProjectMeta object.
        :type project_meta: ProjectMeta
        :param nrrd_paths: Paths to the converted NRRD files from STL.
        :type nrrd_paths: List[str]
        :param key_id_map: The Key to ID map.
        :type key_id_map: KeyIdMap
        :param project_id: The Project ID to update metadata on upload (optional).
        :type project_id: int, optional
        :return: A tuple containing the updated ann and project_meta objects.
        :rtype: Tuple[VolumeAnnotation, ProjectMeta]
        """
        for nrrd_path in nrrd_paths:
            object_key = None

            # searching connection between interpolation and spatial figure in annotations and set its object_key
            for sf in ann.spatial_figures:
                if sf.key().hex == get_file_name(nrrd_path):
                    object_key = sf.parent_object.key()
                    break

            if object_key:
                for obj in ann.objects:
                    if obj.key() == object_key:
                        class_title = obj.obj_class.name
                        break

            else:
                raise Exception(
                    f"Can't find volume object for Mask 3D from unknown file '{nrrd_path}'. Please check the project structure."
                )

            new_obj_class = ObjClass(f"{class_title}_mask_3d", Mask3D)
            project_meta = project_meta.add_obj_class(new_obj_class)

            if transfer_type == "download":
                geometry = Mask3D(np.random.randint(2, size=(3, 3, 3), dtype=np.bool_))
            elif transfer_type == "upload":
                self._api.project.update_meta(project_id, project_meta)
                geometry = Mask3D.from_file(nrrd_path)

            new_object = VolumeObject(new_obj_class, mask_3d=geometry)

            # add new Volume object to VolumeAnnotation with spatial figure
            ann = ann.add_objects([new_object])
            key_id_map.add_object(new_object.key(), id=None)
            key_id_map.add_figure(new_object.figure.key(), id=None)

            if transfer_type == "download":
                os.rename(
                    nrrd_path, f"{os.path.dirname(nrrd_path)}/{new_object.figure.key().hex}.nrrd"
                )
                stl_path = re.sub(r"\.[^.]+$", ".stl", nrrd_path)
                os.rename(
                    stl_path, f"{os.path.dirname(nrrd_path)}/{new_object.figure.key().hex}.stl"
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
        Add new VolumeObjects with spatial figures (Mask3D) to a VolumeAnnotation in the project.

        :param volume_id: The ID of the volume.
        :type volume_id: int
        :param objects: New volume objects with spatial figures (Mask3D).
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

        if isinstance(objects, List):
            objects = VolumeObjectCollection(objects)

        # check if objects without figures
        for _, vobject in objects._collection.items():
            try:
                vobject.figure
            except AttributeError as e:
                e.args = [
                    "3D mask for object is not defined",
                ]
                raise e

        sf_figures = [vobject.figure for vobject in objects]
        volume_meta = self._api.volume.get_info_by_id(volume_id).meta
        ann = VolumeAnnotation(volume_meta, objects, spatial_figures=sf_figures)
        self.append(volume_id, ann, key_id_map)
