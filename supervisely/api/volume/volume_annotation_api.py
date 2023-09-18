# coding: utf-8


from typing import List, Optional, Union, Callable, Dict, Tuple

from tqdm import tqdm
import numpy as np
import os
import random


from supervisely.project.project_meta import ProjectMeta
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation

from supervisely.api.entity_annotation.entity_annotation_api import EntityAnnotationAPI
from supervisely.io.json import load_json_file
from supervisely.io.fs import dir_exists, get_file_name
from supervisely.volume import stl_converter
import supervisely

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

    def append(
        self,
        volume_id: int,
        ann: VolumeAnnotation,
        key_id_map: KeyIdMap = None,
        sf_geometries: Dict = None,
    ):
        """
        Loads VolumeAnnotation to a given volume ID in the API.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param ann: VolumeAnnotation object.
        :type ann: VolumeAnnotation
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :param sf_geometries: Dict where keys are hex of sf.key(), and values are geometries, which represented as NRRD in byte format
        :type key_id_map: Dict, optional
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
        if sf_geometries:
            self._api.volume.figure.upload_sf_geometries(
                ann.spatial_figures, sf_geometries, key_id_map
            )

    def upload_paths(
        self,
        volume_ids: List[int],
        ann_paths: List[str],
        project_meta: ProjectMeta,
        interpolation_dirs=None,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        mask_dirs=None,
    ):
        """
        Loads VolumeAnnotations from a given paths to a given volumes IDs in the API. Volumes IDs must be from one dataset.

        :param volume_ids: Volumes IDs in Supervisely.
        :type volume_ids: List[int]
        :param ann_paths: Paths to annotations on local machine.
        :type ann_paths: List[str]
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>` for VolumeAnnotations.
        :type project_meta: ProjectMeta
        :param interpolation_dirs:
        :type interpolation_dirs:
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
        :param mask_dirs:
        :type mask_dirs:
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

            geometries_dict = {}

            # convert STL interplations to NRRD and prepare to upload as a spatial figures
            if dir_exists(interpolation_dir):
                nrrd_paths = stl_converter.save_to_nrrd_file_on_upload(interpolation_dir)

                ann_json, project_meta, stl_geometries_dict = self.update_project_on_upload(
                    project_id, ann_json, project_meta, nrrd_paths, key_id_map
                )
                geometries_dict.update(stl_geometries_dict)
                geometries_dict.update(
                    self._api.volume.figure.read_sf_geometries(interpolation_dir)
                )

            ann = supervisely.VolumeAnnotation.from_json(ann_json, project_meta)

            # prepare spatial figure geometries
            if dir_exists(mask_dir):
                geometries_dict.update(self._api.volume.figure.read_sf_geometries(mask_dir))

            self.append(volume_id, ann, key_id_map, geometries_dict)

            if progress_cb is not None:
                progress_cb(1)

    def update_project_on_upload(
        self,
        project_id: int,
        ann_json: Dict,
        project_meta: ProjectMeta,
        nrrd_full_paths: List[str],
        key_id_map: KeyIdMap,
    ) -> Tuple[Dict, ProjectMeta, Dict]:
        """
        Creates new ObjClass and VolumeFigure annotations for converted STL and updates project meta.
        Replaces ClosedMeshSurface spatial figures with Mask 3D.
        Read geometries for new figures and store in dictionary.

        :param project_id: Project ID
        :type project_id: int
        :param ann_json: Volume Annotation in JSON format
        :type ann_json: Dict
        :param project_meta: ProjectMeta object
        :type project_meta: ProjectMeta
        :param nrrd_full_paths: Paths for converted NRRD from STL
        :type nrrd_full_paths: List[str]
        :param key_id_map: Key to ID map
        :type key_id_map: KeyIdMap
        :return: Updated ann_json, project_meta and prepared geometries_dict
        :rtype: Tuple[Dict, ProjectMeta, Dict]
        :Usage example:

        """

        obj_classes_list = []
        geometries_dict = {}

        for path in nrrd_full_paths:
            object_key = None

            # searching connection between interpolation and spatial figure in annotations and set its object_key
            for sf in ann_json.get("spatialFigures"):
                if sf.get("key") == get_file_name(path):
                    object_key = sf.get("objectKey")
                    break

            if object_key:
                for obj in ann_json.get("objects"):
                    if obj.get("key") == object_key:
                        class_title = obj.get("classTitle")
                        break
            # if this external interpolation class name generates with the class_title as file name
            else:
                class_title = get_file_name(path)
                sf = None

            new_obj_class = supervisely.ObjClass(
                f"stl_{class_title}_interpolation", supervisely.Mask3D
            )
            obj_classes_list.append(new_obj_class)
            new_object = supervisely.VolumeObject(new_obj_class)

            # add new Volume object to ann_json with dummy figure
            ann_json.get("objects").append(new_object.to_json(key_id_map))
            new_class_figure = supervisely.VolumeFigure(
                new_object, supervisely.Mask3D(np.random.randint(2, size=(3, 3, 3), dtype=np.bool_))
            )

            # add new spatial figure to ann_json
            ann_json.get("spatialFigures").append(new_class_figure.to_json(key_id_map))
            # remove stl spatial figure from ann_json
            if sf:
                ann_json.get("spatialFigures").remove(sf)

            with open(path, "rb") as file:
                geometry_bytes = file.read()
            geometries_dict[new_class_figure.key().hex] = geometry_bytes

        # updates project meta if there are new classes
        if obj_classes_list:
            new_meta = ProjectMeta(obj_classes_list)
            project_meta = project_meta.merge(new_meta)
            self._api.project.update_meta(project_id, project_meta)

        return ann_json, project_meta, geometries_dict

    def update_project_on_download(
        self,
        ann: VolumeAnnotation,
        project_meta: ProjectMeta,
        nrrd_paths: List[str],
        key_id_map: KeyIdMap,
    ) -> Tuple[VolumeAnnotation, ProjectMeta]:
        """
        Creates new ObjClass and VolumeFigure annotations for converted STL.
        Replaces ClosedMeshSurface spatial figures with Mask 3D.
        Updates ann, project meta, key_id_map

        :param ann: VolumeAnnotation object
        :type ann: VolumeAnnotation
        :param project_meta: ProjectMeta object
        :type project_meta: ProjectMeta
        :param nrrd_paths: Paths for converted NRRD from STL
        :type nrrd_paths: List[str]
        :param key_id_map: Key to ID map
        :type key_id_map: KeyIdMap
        :return: Updated ann, project_meta
        :rtype: Tuple[VolumeAnnotation, ProjectMeta]
        :Usage example:
        """
        # create unique ids for new objects
        if nrrd_paths:
            max_f_id = max(key_id_map.to_dict().get("figures").values())
            max_o_id = max(key_id_map.to_dict().get("objects").values())
            new_f_id = max_f_id + (10 ** (len(str(max_f_id)) - 1)) + random.randint(100, 999)
            new_o_id = max_o_id + (10 ** (len(str(max_o_id)) - 1)) + random.randint(100, 999)

        for path in nrrd_paths:
            object_key = None

            # searching connection between interpolation and spatial figure in annotations and set its object_key
            for sf in ann.spatial_figures:
                if sf.key().hex == get_file_name(path):
                    object_key = sf.parent_object.key()
                    break

            if object_key:
                for obj in ann.objects:
                    if obj.key() == object_key:
                        class_title = obj.obj_class.name
                        break
            # if this external interpolation class name generates with the class_title as file name
            else:
                class_title = get_file_name(path)
                sf = None

            new_obj_class = supervisely.ObjClass(
                f"stl_{class_title}_interpolation", supervisely.Mask3D
            )
            project_meta = project_meta.add_obj_class(new_obj_class)
            new_object = supervisely.VolumeObject(new_obj_class)
            new_collection = ann.objects.add(new_object)
            ann = ann.clone(objects=new_collection)
            key_id_map.add_object(new_object.key(), id=new_o_id)

            # add new Volume object to ann with dummy geometry
            new_class_figure = supervisely.VolumeFigure(
                new_object, supervisely.Mask3D(np.random.randint(2, size=(3, 3, 3), dtype=np.bool_))
            )

            # add new spatial figure to ann
            ann.spatial_figures.append(new_class_figure)
            key_id_map.add_figure(new_class_figure.key(), id=new_f_id)

            os.rename(path, f"{os.path.dirname(path)}/{new_class_figure.key().hex}.nrrd")

            # remove stl spatial figure from ann
            if sf:
                ann.spatial_figures.remove(sf)

        return ann, project_meta
