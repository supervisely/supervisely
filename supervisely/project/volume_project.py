# coding: utf-8
import os
import re
import sys
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy
from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.api import Api
from supervisely.api.module_api import ApiField
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import change_directory_at_index, touch
from supervisely.project.project import OpenMode
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import VideoDataset, VideoProject
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress, tqdm_sly
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume import stl_converter
from supervisely.volume import volume as sly_volume
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_figure import VolumeFigure

VolumeItemPaths = namedtuple("VolumeItemPaths", ["volume_path", "ann_path"])


class VolumeDataset(VideoDataset):
    item_dir_name = "volume"
    interpolation_dir = "interpolation"
    interpolation_dir_name = interpolation_dir
    mask_dir = "mask"
    mask_dir_name = mask_dir
    annotation_class = VolumeAnnotation
    item_module = sly_volume
    paths_tuple = VolumeItemPaths

    @classmethod
    def _has_valid_ext(cls, path: str) -> bool:
        """
        Checks if file from given path is supported
        :param path: str
        :return: bool
        """
        return sly_volume.has_valid_ext(path)

    def _get_empty_annotaion(self, item_name):
        path = item_name
        _, volume_meta = sly_volume.read_nrrd_serie_volume(path)
        return self.annotation_class(volume_meta)

    def get_interpolation_dir(self, item_name):
        return os.path.join(self.directory, self.interpolation_dir, item_name)

    def get_interpolation_path(self, item_name, figure):
        return os.path.join(self.get_interpolation_dir(item_name), figure.key().hex + ".stl")

    def get_mask_dir(self, item_name):
        return os.path.join(self.directory, self.mask_dir, item_name)

    def get_mask_path(self, item_name, figure):
        return os.path.join(self.get_mask_dir(item_name), figure.key().hex + ".nrrd")

    def get_classes_stats(
        self,
        project_meta: Optional[ProjectMeta] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        if project_meta is None:
            project = VolumeProject(self.project_dir, OpenMode.READ)
            project_meta = project.meta
        class_items = {}
        class_objects = {}
        class_figures = {}
        for obj_class in project_meta.obj_classes:
            class_items[obj_class.name] = 0
            class_objects[obj_class.name] = 0
            class_figures[obj_class.name] = 0
        for item_name in self:
            item_ann = self.get_ann(item_name, project_meta)
            item_class = {}
            for ann_obj in item_ann.objects:
                class_objects[ann_obj.obj_class.name] += 1
            for volume_figure in item_ann.figures:
                class_figures[volume_figure.parent_object.obj_class.name] += 1
                item_class[volume_figure.parent_object.obj_class.name] = True
            for obj_class in project_meta.obj_classes:
                if obj_class.name in item_class.keys():
                    class_items[obj_class.name] += 1

        result = {}
        if return_items_count:
            result["items_count"] = class_items
        if return_objects_count:
            result["objects_count"] = class_objects
        if return_figures_count:
            result["figures_count"] = class_figures
        return result


class VolumeProject(VideoProject):
    dataset_class = VolumeDataset

    class DatasetDict(KeyIndexedCollection):
        item_type = VolumeDataset

    def get_classes_stats(
        self,
        dataset_names: Optional[List[str]] = None,
        return_objects_count: Optional[bool] = True,
        return_figures_count: Optional[bool] = True,
        return_items_count: Optional[bool] = True,
    ):
        return super(VolumeProject, self).get_classes_stats(
            dataset_names, return_objects_count, return_figures_count, return_items_count
        )

    @property
    def type(self) -> str:
        """
        Project type.

        :return: Project type.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            project = sly.VolumeProject("/home/admin/work/supervisely/projects/volumes", sly.OpenMode.READ)
            print(project.type)
            # Output: 'volumes'
        """
        return ProjectType.VOLUMES.value

    @staticmethod
    def download(
        api: Api,
        project_id: int,
        dest_dir: str,
        dataset_ids: Optional[List[int]] = None,
        download_volumes: Optional[bool] = True,
        log_progress: bool = False,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
        **kwargs,
    ) -> None:
        """
        Download volume project from Supervisely to the given directory.

        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param project_id: Supervisely downloadable project ID.
        :type project_id: :class:`int`
        :param dest_dir: Destination directory.
        :type dest_dir: :class:`str`
        :param dataset_ids: Dataset IDs.
        :type dataset_ids: :class:`list` [ :class:`int` ], optional
        :param download_volumes: Download volume data files or not.
        :type download_volumes: :class:`bool`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: bool
        :param progress_cb: Function for tracking the download progress.
        :type progress_cb: tqdm or callable, optional

        :return: None
        :rtype: NoneType
        :Usage example:
        .. code-block:: python

                import supervisely as sly

                # Local destination Volume Project folder
                save_directory = "/home/admin/work/supervisely/source/vlm_project"

                # Obtain server address and your api_token from environment variables
                # Edit those values if you run this notebook on your own PC
                address = os.environ['SERVER_ADDRESS']
                token = os.environ['API_TOKEN']

                # Initialize API object
                api = sly.Api(address, token)
                project_id = 8888

                # Download Project
                sly.VolumeProject.download(api, project_id, save_directory)
                project_fs = sly.VolumeProject(save_directory, sly.OpenMode.READ)
        """
        download_volume_project(
            api=api,
            project_id=project_id,
            dest_dir=dest_dir,
            dataset_ids=dataset_ids,
            download_volumes=download_volumes,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def upload(
        directory: str,
        api: Api,
        workspace_id: int,
        project_name: Optional[str] = None,
        log_progress: bool = True,
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> Tuple[int, str]:
        """
        Uploads volume project to Supervisely from the given directory.

        :param directory: Path to project directory.
        :type directory: :class:`str`
        :param api: Supervisely API address and token.
        :type api: :class:`Api<supervisely.api.api.Api>`
        :param workspace_id: Workspace ID, where project will be uploaded.
        :type workspace_id: :class:`int`
        :param project_name: Name of the project in Supervisely. Can be changed if project with the same name is already exists.
        :type project_name: :class:`str`, optional
        :param log_progress: Show uploading progress bar.
        :type log_progress: :class:`bool`
        :param progress_cb: Function for tracking the download progress.
        :type progress_cb: tqdm or callable, optional

        :return: Project ID and name. It is recommended to check that returned project name coincides with provided project name.
        :rtype: :class:`int`, :class:`str`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            # Local folder with Volume Project
            project_directory = "/home/admin/work/supervisely/source/vlm_project"

            # Obtain server address and your api_token from environment variables
            # Edit those values if you run this notebook on your own PC
            address = os.environ['SERVER_ADDRESS']
            token = os.environ['API_TOKEN']

            # Initialize API object
            api = sly.Api(address, token)

            # Upload Volume Project
            project_id, project_name = sly.VolumeProject.upload(
                project_directory,
                api,
                workspace_id=45,
                project_name="My Volume Project"
            )
        """
        return upload_volume_project(
            dir=directory,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )

    @staticmethod
    def get_train_val_splits_by_count(project_dir: str, train_count: int, val_count: int) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_count()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_tag(
        project_dir: str,
        train_tag_name: str,
        val_tag_name: str,
        untagged: Optional[str] = "ignore",
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_dataset(
        project_dir: str, train_datasets: List[str], val_datasets: List[str]
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_tag()' is not supported for VolumeProject class now."
        )

    @staticmethod
    def get_train_val_splits_by_collections(
        project_dir: str,
        train_collections: List[int],
        val_collections: List[int],
        project_id: int,
        api: Api,
    ) -> None:
        """
        Not available for VolumeProject class.
        :raises: :class:`NotImplementedError` in all cases.
        """
        raise NotImplementedError(
            f"Static method 'get_train_val_splits_by_collections()' is not supported for VolumeProject class now."
        )

    @staticmethod
    async def download_async(*args, **kwargs):
        raise NotImplementedError(
            f"Static method 'download_async()' is not supported for VolumeProject class now."
        )


def download_volume_project(
    api: Api,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    download_volumes: Optional[bool] = True,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> None:
    """
    Download volume project to the local directory.

    :param api: Supervisely API address and token.
    :type api: Api
    :param project_id: Project ID to download.
    :type project_id: int
    :param dest_dir: Destination path to local directory.
    :type dest_dir: str
    :param dataset_ids: Specified list of Dataset IDs which will be downloaded. Datasets could be downloaded from different projects but with the same data type.
    :type dataset_ids: list(int), optional
    :param download_volumes: Include volumes in the download.
    :type download_volumes: bool, optional
    :param log_progress: Show downloading logs in the output.
    :type log_progress: bool
    :param progress_cb: Function for tracking download progress.
    :type progress_cb: tqdm or callable, optional

    :return: None.
    :rtype: NoneType
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        from tqdm import tqdm
        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        dest_dir = 'your/local/dest/dir'

        # Download volume project
        project_id = 18532
        project_info = api.project.get_info_by_id(project_id)
        num_volumes = project_info.items_count

        p = tqdm(desc="Downloading volume project", total=num_volumes)
        sly.download_volume_project(
            api,
            project_id,
            dest_dir,
            progress_cb=p,
        )
    """

    LOG_BATCH_SIZE = 1

    key_id_map = KeyIdMap()

    project_fs = VolumeProject(dest_dir, OpenMode.CREATE)

    meta = ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    datasets_infos = []
    if dataset_ids is not None:
        for ds_id in dataset_ids:
            datasets_infos.append(api.dataset.get_info_by_id(ds_id))
    else:
        datasets_infos = api.dataset.get_list(project_id)

    for dataset in datasets_infos:
        dataset_fs: VolumeDataset = project_fs.create_dataset(dataset.name)
        volumes = api.volume.get_list(dataset.id)

        ds_progress = progress_cb
        if log_progress:
            ds_progress = tqdm_sly(
                desc="Downloading volumes from: {!r}".format(dataset.name),
                total=len(volumes),
            )
        for batch in batched(volumes, batch_size=LOG_BATCH_SIZE):
            volume_ids = [volume_info.id for volume_info in batch]
            volume_names = [volume_info.name for volume_info in batch]

            ann_jsons = api.volume.annotation.download_bulk(dataset.id, volume_ids)

            for volume_id, volume_name, volume_info, ann_json in zip(
                volume_ids, volume_names, batch, ann_jsons
            ):
                if volume_name != ann_json[ApiField.VOLUME_NAME]:
                    raise RuntimeError(
                        "Error in api.volume.annotation.download_batch: broken order"
                    )
                try:
                    ann = VolumeAnnotation.from_json(ann_json, project_fs.meta, key_id_map)
                except Exception as e:
                    logger.info(
                        "INFO FOR DEBUGGING",
                        extra={
                            "project_id": project_id,
                            "dataset_id": dataset.id,
                            "volume_id": volume_id,
                            "volume_name": volume_name,
                            "ann_json": ann_json,
                        },
                    )
                    raise e

                volume_file_path = dataset_fs.generate_item_path(volume_name)
                if download_volumes is True:
                    header = None
                    item_progress = None
                    if ds_progress is not None:
                        item_progress = tqdm_sly(
                            desc=f"Downloading '{volume_name}'",
                            total=volume_info.sizeb,
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        )
                        api.video.download_path(volume_id, volume_file_path, item_progress)
                    else:
                        api.volume.download_path(volume_id, volume_file_path)
                else:
                    touch(volume_file_path)
                    header = _create_volume_header(ann)

                mask_ids = []
                mask_paths = []
                mesh_ids = []
                mesh_paths = []
                for sf in ann.spatial_figures:
                    figure_id = key_id_map.get_figure_id(sf.key())
                    if sf.geometry.name() == Mask3D.name():
                        mask_ids.append(figure_id)
                        figure_path = dataset_fs.get_mask_path(volume_name, sf)
                        mask_paths.append(figure_path)
                    if sf.geometry.name() == ClosedSurfaceMesh.name():
                        mesh_ids.append(figure_id)
                        figure_path = dataset_fs.get_interpolation_path(volume_name, sf)
                        mesh_paths.append(figure_path)
                
                figs = api.volume.figure.download(dataset.id, [volume_id], skip_geometry=True)
                figs = figs.get(volume_id, {})
                figs_ids_map = {fig.id: fig for fig in figs}
                for ann_fig in ann.figures + ann.spatial_figures:
                    fig = figs_ids_map.get(ann_fig.geometry.sly_id)
                    ann_fig.custom_data.update(fig.custom_data)

                api.volume.figure.download_stl_meshes(mesh_ids, mesh_paths)
                api.volume.figure.download_sf_geometries(mask_ids, mask_paths)

                # prepare a list of paths where converted STLs will be stored
                nrrd_paths = []
                for file in mesh_paths:
                    file = re.sub(r"\.[^.]+$", ".nrrd", file)
                    file = change_directory_at_index(file, "mask", -3)  # change destination folder
                    nrrd_paths.append(file)

                stl_converter.to_nrrd(mesh_paths, nrrd_paths, header=header)

                ann, meta = api.volume.annotation._update_on_transfer(
                    "download", ann, project_fs.meta, nrrd_paths
                )

                project_fs.set_meta(meta)

                dataset_fs.add_item_file(
                    volume_name,
                    volume_file_path,
                    ann=ann,
                    _validate_item=False,
                )

                if progress_cb is not None:
                    progress_cb(1)

            if log_progress:
                ds_progress(len(batch))

    project_fs.set_key_id_map(key_id_map)


def load_figure_data(
    api: Api, volume_file_path: str, spatial_figure: VolumeFigure, key_id_map: KeyIdMap
):
    """
    Load data into figure geometry.

    :param api: Supervisely API address and token.
    :type api: Api
    :param volume_file_path: Path to Volume file location
    :type volume_file_path: str
    :param spatial_figure: Spatial figure
    :type spatial_figure: VolumeFigure object
    :param key_id_map: Mapped keys and IDs
    :type key_id_map: KeyIdMap object
    """
    figure_id = key_id_map.get_figure_id(spatial_figure.key())
    figure_path = "{}_mask3d/".format(volume_file_path[:-5]) + f"{figure_id}.nrrd"
    api.volume.figure.download_stl_meshes([figure_id], [figure_path])
    Mask3D.from_file(spatial_figure, figure_path)


# TODO: add methods to convert to 3d masks


def upload_volume_project(
    dir: str,
    api: Api,
    workspace_id: int,
    project_name: Optional[str] = None,
    log_progress: bool = True,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
) -> Tuple[int, str]:
    project_fs = VolumeProject.read_single(dir)
    if project_name is None:
        project_name = project_fs.name

    if api.project.exists(workspace_id, project_name):
        project_name = api.project.get_free_name(workspace_id, project_name)

    project = api.project.create(workspace_id, project_name, ProjectType.VOLUMES)
    api.project.update_meta(project.id, project_fs.meta.to_json())

    if progress_cb is not None:
        log_progress = False

    item_id_dct, anns_paths_dct, interpolation_dirs_dct, mask_dirs_dct = {}, {}, {}, {}

    for dataset_fs in project_fs.datasets:
        dataset_fs: VolumeDataset
        dataset = api.dataset.create(project.id, dataset_fs.name)

        names, item_paths, ann_paths, mask_dirs, interpolation_dirs = [], [], [], [], []
        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            names.append(item_name)
            item_paths.append(img_path)
            ann_paths.append(ann_path)
            interpolation_dirs.append(dataset_fs.get_interpolation_dir(item_name))
            mask_dirs.append(dataset_fs.get_mask_dir(item_name))

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm_sly(
                desc="Uploading volumes to {!r}".format(dataset.name),
                total=len(item_paths),
                position=0,
            )

        item_infos = api.volume.upload_nrrd_series_paths(
            dataset.id, names, item_paths, ds_progress, log_progress
        )
        volume_ids = [item_info.id for item_info in item_infos]

        anns_progress = None
        if log_progress is True or progress_cb is not None:
            anns_progress = tqdm_sly(
                desc="Uploading annotations to {!r}".format(dataset.name),
                total=len(volume_ids),
                leave=False,
            )
        api.volume.annotation.upload_paths(
            volume_ids,
            ann_paths,
            project_fs.meta,
            interpolation_dirs,
            anns_progress,
            mask_dirs,
        )

    return project.id, project.name


def _create_volume_header(ann: VolumeAnnotation) -> Dict:
    """
    Create volume header to use in STL converter when downloading project without volumes.

    :param ann: VolumeAnnotation object
    :type ann: VolumeAnnotation
    :return: header with Volume meta parameters
    :rtype: Dict
    """
    header = {}
    header["sizes"] = numpy.array([value for _, value in ann.volume_meta["dimensionsIJK"].items()])
    world_matrix = ann.volume_meta["IJK2WorldMatrix"]
    header["space directions"] = numpy.array(
        [world_matrix[i : i + 3] for i in range(0, len(world_matrix) - 4, 4)]
    )
    header["space origin"] = numpy.array(
        [world_matrix[i + 3] for i in range(0, len(world_matrix) - 4, 4)]
    )
    if ann.volume_meta["ACS"] == "RAS":
        header["space"] = "right-anterior-superior"
    elif ann.volume_meta["ACS"] == "LAS":
        header["space"] = "left-anterior-superior"
    elif ann.volume_meta["ACS"] == "LPS":
        header["space"] = "left-posterior-superior"
    return header
