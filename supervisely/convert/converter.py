import os

from tqdm import tqdm
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from supervisely._utils import is_production
from supervisely.api.api import Api
from supervisely.app import get_data_dir
from supervisely.convert.image.csv.csv_converter import CSVConverter
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
    unpack_archive,
)
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress


class ImportManager:
    def __init__(
        self,
        input_data: str,
        project_type: ProjectType,
        team_id: int = None,
        labeling_interface: Literal[
            "default",
            "multi_view",
            "multispectral",
            "images_with_16_color",
            "medical_imaging_single",
        ] = "default",
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

        self._input_data = self._prepare_input_data(input_data)
        self._unpack_archives(self._input_data)
        remove_junk_from_dir(self._input_data)

        self._modality = project_type
        self._converter = self.get_converter()
        if isinstance(self._converter, CSVConverter):
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
        if str(self._modality) == ProjectType.IMAGES.value:
            return ImageConverter(self._input_data, self._labeling_interface)._converter
        elif str(self._modality) == ProjectType.VIDEOS.value:
            return VideoConverter(self._input_data, self._labeling_interface)._converter
        elif str(self._modality) == ProjectType.POINT_CLOUDS.value:
            return PointcloudConverter(self._input_data, self._labeling_interface)._converter
        elif str(self.modality) == ProjectType.VOLUMES.value:
            return VolumeConverter(self._input_data, self._labeling_interface)._converter
        elif str(self._modality) == ProjectType.POINT_CLOUD_EPISODES.value:
            return PointcloudEpisodeConverter(self._input_data, self._labeling_interface)._converter
        else:
            raise ValueError(f"Unsupported project type selected: {self._modality}")

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        self.converter.upload_dataset(self._api, dataset_id)

    # def validate_format(self):
    #     raise NotImplementedError

    def _prepare_input_data(self, input_data):
        if dir_exists(input_data):
            logger.info(f"Input data is a local directory: {input_data}")
            return input_data
        elif file_exists(input_data):
            logger.info(f"Input data is a local file: {input_data}. Will use its directory")
            return os.path.dirname(input_data)
        elif self._api.storage.exists(self._team_id, input_data):
            logger.info(f"Input data is a remote file: {input_data}")
            return self._download_input_data(input_data)
        elif self._api.storage.dir_exists(self._team_id, input_data):
            logger.info(f"Input data is a remote directory: {input_data}")
            return self._download_input_data(input_data, is_dir=True)
        else:
            raise RuntimeError(f"Input data not found: {input_data}")

    def _download_input_data(self, remote_path, is_dir=False):
        """Download input data from Supervisely"""

        if not is_dir:
            dir_name = "Import data"
            local_path = os.path.join(get_data_dir(), dir_name)
            mkdir(local_path, remove_content_if_exists=True)
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

    def _unpack_archives(self, local_path):
        """Unpack if input data contains an archive."""

        new_paths_to_scan = [local_path]
        while len(new_paths_to_scan) > 0:
            archives = []
            path = new_paths_to_scan.pop()
            for root, _, files in os.walk(path):
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
