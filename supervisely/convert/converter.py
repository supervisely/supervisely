import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely import fs
from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from supervisely.convert.image.csv.csv_converter import CSVConverter
from supervisely.convert.image.high_color.high_color_depth import (
    HighColorDepthImageConverter,
)
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
    PointcloudEpisodeConverter,
)
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.io.env import team_id as env_team_id
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    is_archive,
    mkdir,
    remove_junk_from_dir,
    silent_remove,
    touch,
    unpack_archive,
)
from supervisely.project.project import Project
from supervisely.project.project import Dataset
from supervisely import OpenMode, ProjectMeta
from supervisely.project.project_settings import LabelingInterface
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class ImportManager:

    def __init__(
        self,
        input_data: Union[str, List[str]],
        project_type: ProjectType,
        team_id: Optional[int] = None,
        labeling_interface: LabelingInterface = LabelingInterface.DEFAULT,
        upload_as_links: bool = False,
    ):
        self._api = Api.from_env()
        if team_id is not None:
            team_info = self._api.team.get_info_by_id(team_id)
            if team_info is None:
                raise ValueError(
                    f"Team with id {team_id} does not exist or you do not have access to it."
                )
        else:
            self._team_id = env_team_id()
        self._labeling_interface = labeling_interface
        self._upload_as_links = upload_as_links
        self._remote_files_map = {}
        self._modality = project_type

        if isinstance(input_data, str):
            input_data = [input_data]

        self._input_data = get_data_dir()
        for data in input_data:
            self._prepare_input_data(data)
        self._unpack_archives(self._input_data)
        remove_junk_from_dir(self._input_data)

        self._converter = self.get_converter()
        if isinstance(self._converter, (HighColorDepthImageConverter, CSVConverter)):
            self._converter.team_id = self._team_id

    @property
    def modality(self):
        return self._modality

    @property
    def converter(self):
        return self._converter

    def get_items(self):
        return self._converter.get_items()

    def get_converter(self):
        """Return correct converter"""
        modality_converter_map = {
            ProjectType.IMAGES.value: ImageConverter,
            ProjectType.VIDEOS.value: VideoConverter,
            ProjectType.POINT_CLOUDS.value: PointcloudConverter,
            ProjectType.VOLUMES.value: VolumeConverter,
            ProjectType.POINT_CLOUD_EPISODES.value: PointcloudEpisodeConverter,
        }
        if str(self._modality) not in modality_converter_map:
            raise ValueError(f"Unsupported project type selected: {self._modality}")

        modality_converter = modality_converter_map[str(self._modality)](
            self._input_data,
            self._labeling_interface,
            self._upload_as_links,
            self._remote_files_map,
        )
        return modality_converter.detect_format()

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        self.converter.upload_dataset(self._api, dataset_id)

    # def validate_format(self):
    #     raise NotImplementedError

    def _prepare_input_data(self, input_data):
        logger.debug(f"Preparing input data: {input_data}")
        if dir_exists(input_data):
            logger.info(f"Input data is a local directory: {input_data}")
            # return input_data
            dst_dir = os.path.join(get_data_dir(), os.path.basename(os.path.normpath(input_data)))
            fs.copy_dir_recursively(input_data, dst_dir)
        elif file_exists(input_data):
            logger.info(f"Input data is a local file: {input_data}. Will use its directory")
            # return os.path.dirname(input_data)
            dst_file = os.path.join(get_data_dir(), os.path.basename(input_data))
            fs.copy_file(input_data, dst_file)
        elif self._api.storage.exists(self._team_id, input_data):
            if self._upload_as_links and str(self._modality) in [
                ProjectType.IMAGES.value,
                ProjectType.VIDEOS.value,
            ]:
                logger.info(f"Input data is a remote file: {input_data}. Scanning...")
                return self._reproduce_remote_files(input_data)
            else:
                if self._upload_as_links and str(self._modality) == ProjectType.VOLUMES.value:
                    self._scan_remote_files(input_data)
                logger.info(f"Input data is a remote file: {input_data}. Downloading...")
                return self._download_input_data(input_data)
        elif self._api.storage.dir_exists(self._team_id, input_data):
            if self._upload_as_links and str(self._modality) in [
                ProjectType.IMAGES.value,
                ProjectType.VIDEOS.value,
            ]:
                logger.info(f"Input data is a remote directory: {input_data}. Scanning...")
                return self._reproduce_remote_files(input_data, is_dir=True)
            else:
                if self._upload_as_links and str(self._modality) == ProjectType.VOLUMES.value:
                    self._scan_remote_files(input_data, is_dir=True)
                logger.info(f"Input data is a remote directory: {input_data}. Downloading...")
                return self._download_input_data(input_data, is_dir=True)
        else:
            raise RuntimeError(f"Input data not found: {input_data}")

    def _download_input_data(self, remote_path, is_dir=False):
        """Download input data from Supervisely"""

        if not is_dir:
            dir_name = "Import data"
            local_path = os.path.join(get_data_dir(), dir_name)
            mkdir(local_path, remove_content_if_exists=False)
            save_path = os.path.join(local_path, os.path.basename(remote_path))
        else:
            dir_name = os.path.basename(os.path.normpath(remote_path))
            local_path = os.path.join(get_data_dir(), dir_name)

        if not is_dir:
            files_size = self._api.storage.get_info_by_path(self._team_id, remote_path).sizeb
            progress, progress_cb = self._get_progress(files_size)
            self._api.storage.download(
                self._team_id, remote_path, save_path, progress_cb=progress_cb
            )
            if not is_production():
                progress.close()
        else:
            directory_size = self._api.storage.get_directory_size(self._team_id, remote_path)
            progress, progress_cb = self._get_progress(directory_size)
            self._api.storage.download_directory(
                self._team_id, remote_path, local_path, progress_cb=progress_cb
            )
            if not is_production():
                progress.close()

        return local_path

    def _scan_remote_files(self, remote_path, is_dir=False):
        """
        Scan remote directory. Collect local-remote paths mapping
        Will be used to save relations between uploaded files and remote files (for volumes).
        """

        dir_path = remote_path.rstrip("/") if is_dir else os.path.dirname(remote_path)
        dir_name = os.path.basename(dir_path)

        local_path = os.path.join(get_data_dir(), dir_name)

        if is_dir:
            files = self._api.storage.list(self._team_id, remote_path, include_folders=False)
        else:
            files = [self._api.storage.get_info_by_path(self._team_id, remote_path)]

        unique_directories = set()
        for file in files:
            new_path = file.path.replace(dir_path, local_path)
            self._remote_files_map[new_path] = file.path
            unique_directories.add(str(Path(file.path).parent))

        logger.info(f"Scanned remote directories:\n   - " + "\n   - ".join(unique_directories))
        return local_path

    def _reproduce_remote_files(self, remote_path, is_dir=False):
        """
        Scan remote directory and create dummy structure locally.
        Will be used to detect annotation format (by dataset structure) remotely.
        """

        dir_path = remote_path.rstrip("/") if is_dir else os.path.dirname(remote_path)
        dir_name = os.path.basename(dir_path)

        local_path = os.path.abspath(os.path.join(get_data_dir(), dir_name))
        mkdir(local_path, remove_content_if_exists=True)

        if is_dir:
            files = self._api.storage.list(self._team_id, remote_path, include_folders=False)
        else:
            files = [self._api.storage.get_info_by_path(self._team_id, remote_path)]

        unique_directories = set()
        for file in files:
            new_path = file.path.replace(dir_path, local_path)
            self._remote_files_map[new_path] = file.path
            Path(new_path).parent.mkdir(parents=True, exist_ok=True)
            unique_directories.add(str(Path(file.path).parent))
            touch(new_path)

        logger.info(f"Scanned remote directories:\n   - " + "\n   - ".join(unique_directories))
        return local_path

    def _unpack_archives(self, local_path):
        """Unpack if input data contains an archive."""

        if self._upload_as_links:
            return
        new_paths_to_scan = [local_path]
        while len(new_paths_to_scan) > 0:
            archives = []
            path = new_paths_to_scan.pop()
            for root, _, files in os.walk(path):
                if Path(root).name == Project.blob_dir_name:
                    logger.info(f"Skip unpacking archive in blob dir: {root}")
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_archive(file_path=file_path):
                        try:
                            new_path = file_path.replace("".join(Path(file_path).suffixes), "")
                            unpack_archive(file_path, new_path)
                            archives.append(file_path)
                            new_paths_to_scan.append(new_path)
                        except Exception as e:
                            logger.error(f"Error while unpacking '{file}': {repr(e)}")

            for archive in archives:
                silent_remove(archive)

    def _get_progress(
        self,
        total: int,
        message: str = "Downloading...",
        is_size: bool = True,
    ) -> tuple:
        if is_production():
            progress = Progress(message, total, is_size=is_size)
            progress_cb = progress.iters_done_report
        else:
            progress = tqdm(
                total=total, desc=message, unit="B" if is_size else "it", unit_scale=is_size
            )
            progress_cb = progress.update
        return progress, progress_cb


class ExportManager:
    def __init__(self, api: Optional[Api] = None):
        self._api = api or Api.from_env()

    def _prepare_local_sly_from_project(
        self, project_id: int, dataset_ids: Optional[List[int]] = None, save_images: bool = True
    ) -> Tuple[Project, ProjectMeta]:
        """Download selected project/datasets to a local SLY folder and return fs objects."""
        export_root = os.path.join(get_data_dir(), "exports")
        mkdir(export_root, remove_content_if_exists=False)
        local_dir = os.path.join(export_root, f"project_{project_id}")
        mkdir(local_dir, remove_content_if_exists=True)

        Project.download(
            api=self._api,
            project_id=project_id,
            dest_dir=local_dir,
            dataset_ids=dataset_ids,
            log_progress=True,
            save_images=save_images,
        )
        project_fs = Project(local_dir, OpenMode.READ)
        return project_fs, project_fs.meta

    def _prepare_local_sly_from_dataset(
        self, dataset_id: int, save_images: bool = True
    ) -> Tuple[Dataset, ProjectMeta]:
        """Download single dataset to a local SLY project folder and return fs objects."""
        ds_info = self._api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_fs, _ = self._prepare_local_sly_from_project(
            ds_info.project_id, [dataset_id], save_images
        )
        dataset_fs = project_fs.datasets.get(ds_info.name)
        return dataset_fs, project_fs.meta

    # def get_available_formats(self, project_id: int) -> Dict[str, bool]:
    #     """Lightweight validation of export formats by ProjectMeta."""
    #     meta_json = self._api.project.get_meta(project_id, with_settings=True)
    #     meta = ProjectMeta.from_json(meta_json)
    #     geom_names = {obj_class.geometry_type.geometry_name() for obj_class in meta.obj_classes}

    #     has_bboxes = "rectangle" in geom_names
    #     has_polygons = "polygon" in geom_names
    #     has_bitmaps = "bitmap" in geom_names
    #     has_keypoints = "graph_nodes" in geom_names

    #     return {
    #         "supervisely": True,
    #         "yolo_detect": has_bboxes,
    #         "yolo_segment": has_polygons or has_bitmaps,
    #         "yolo_pose": has_keypoints,
    #         "coco": has_bboxes or has_polygons or has_bitmaps or has_keypoints,
    #         "coco_keypoints": has_keypoints,
    #         "pascal_voc": has_bboxes or has_polygons or has_bitmaps,
    #     }

    def from_supervisely(
        self,
        *,
        format: str,
        dest_dir: str,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        dataset_ids: Optional[List[int]] = None,
        save_images: bool = True,
        only_annotated: Optional[bool] = None,
        with_annotations: bool = True,
        yolo_task: str = "detect",
        coco_with_captions: bool = False,
        val_datasets: Optional[List[str]] = None,
        is_val: Optional[bool] = None,
        log_progress: bool = True,
        upload_to: Optional[str] = None,
        remote_dir: Optional[str] = None,
        team_id: Optional[int] = None,
        change_name_if_conflict: bool = True,
    ) -> str:
        """
        Entry point for Export Wizard backend: prepare local SLY data and dispatch to target format.

        Parameters are intentionally high-level and map to actual converter options inside.
        """
        mkdir(dest_dir, remove_content_if_exists=False)

        input_data: Union[Project, Dataset]
        meta: ProjectMeta

        if project_id is not None:
            input_data, meta = self._prepare_local_sly_from_project(
                project_id, dataset_ids, save_images
            )
        elif dataset_id is not None:
            input_data, meta = self._prepare_local_sly_from_dataset(dataset_id, save_images)
        else:
            raise ValueError("Either project_id or dataset_id must be provided for export.")

        if not with_annotations:
            return str(Path(input_data.path))

        format_lower = format.lower()
        if format_lower.startswith("yolo"):
            from supervisely.convert.image.yolo.yolo_helper import to_yolo

            return str(
                to_yolo(
                    input_data=input_data,
                    dest_dir=dest_dir,
                    task_type=yolo_task,  # detect/segment/pose
                    meta=meta if isinstance(input_data, Dataset) else None,
                    log_progress=log_progress,
                    val_datasets=val_datasets,
                    is_val=is_val,
                )
            )

        if format_lower.startswith("coco"):
            from supervisely.convert.image.coco.coco_helper import to_coco

            return_dir = to_coco(
                input_data=input_data,
                dest_dir=dest_dir,
                meta=meta if isinstance(input_data, Dataset) else None,
                copy_images=True,
                with_captions=coco_with_captions,
                log_progress=log_progress,
            )
            return str(return_dir) if return_dir is not None else dest_dir

        if format_lower in ("pascal", "pascal_voc", "voc"):
            from supervisely.convert.image.pascal_voc.pascal_voc_helper import to_pascal_voc

            return_dir = to_pascal_voc(
                input_data=input_data,
                dest_dir=dest_dir,
                meta=meta if isinstance(input_data, Dataset) else None,
            )
            return str(return_dir) if return_dir is not None else dest_dir

        if format_lower in ("supervisely", "sly"):
            result_path = str(Path(input_data.path))
        else:
            result_path = dest_dir

        self._upload_destination(
            local_dir=result_path,
            upload_to=upload_to,
            remote_dir=remote_dir,
            team_id=team_id or env_team_id(),
            change_name_if_conflict=change_name_if_conflict,
        )
        return result_path

    def _upload_destination(
        self,
        *,
        local_dir: str,
        upload_to: str,
        remote_dir: str,
        team_id: int,
        change_name_if_conflict: bool,
    ) -> str:
        """Upload a local directory to Team Files or Cloud Storage."""
        upload_to_lower = upload_to.lower()
        # @TODO: archive and upload file
        if upload_to_lower in ("team_files", "teamfiles", "files", "file"):
            return self._api.storage.upload_directory(
                team_id=team_id,
                local_dir=local_dir,
                remote_dir=remote_dir,
                change_name_if_conflict=change_name_if_conflict,
            )
        if upload_to_lower in ("cloud_storage", "cloud", "storage"):
            return self._api.storage.upload_directory(
                team_id=team_id,
                local_dir=local_dir,
                remote_dir=remote_dir,
                change_name_if_conflict=change_name_if_conflict,
            )
        raise ValueError(f"Unsupported upload destination: {upload_to}")
