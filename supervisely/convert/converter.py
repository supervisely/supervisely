import os
from tqdm import tqdm
from supervisely import Api, logger, ProjectType
from supervisely.app import get_data_dir
from supervisely.io.env import team_id as env_team_id
from supervisely.io.fs import is_archive, silent_remove, unpack_archive, dir_exists
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.video.video_converter import VideoConverter
from supervisely.convert.volume.volume_converter import VolumeConverter
from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import PointcloudEpisodeConverter


class ImportManager:
    def __init__(self, input_data: str, project_type: ProjectType, team_id: int = None):
        self._api = Api.from_env()
        if team_id is not None:
            team_info = self._api.team.get_info_by_id(team_id)
            if team_info is None:
                raise ValueError(f"Team with id {team_id} does not exist or you do not have access to it.")
        else:
            self._team_id = env_team_id()


        if dir_exists(input_data):
            logger.info(f"Input data is a local directory: {input_data}")
            self._input_data = input_data
        elif self._api.storage.dir_exists(self._team_id, input_data):
            logger.info(f"Input data is a remote directory: {input_data}")
            self._input_data = self._download_input_data(input_data)
        else:
            raise RuntimeError(f"Input data not found: {input_data}")
        self._unpack_archives(self._input_data)
        self._modality = project_type
        self._converter = self.get_converter()

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
            return ImageConverter(self._input_data)._converter
        elif str(self._modality) == ProjectType.VIDEOS.value:
            return VideoConverter(self._input_data)._converter
        elif str(self._modality) == ProjectType.POINT_CLOUDS.value:
            return PointcloudConverter(self._input_data)._converter
        elif str(self.modality) == ProjectType.VOLUMES.value:
            return VolumeConverter(self._input_data)._converter
        elif str(self._modality) == ProjectType.POINT_CLOUD_EPISODES.value:
            return PointcloudEpisodeConverter(self._input_data)._converter
        else:
            raise ValueError(f"Unsupported project type selected: {self._modality}")

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        self.converter.upload_dataset(self._api, dataset_id)

    # def validate_format(self):
    #     raise NotImplementedError

    def _download_input_data(self, remote_path):
        """Download input data from Supervisely"""

        dir_name = os.path.basename(os.path.normpath(remote_path))
        local_path = os.path.join(get_data_dir(), dir_name)
        directory_size= self._api.storage.get_directory_size(self._team_id, remote_path)
        progress_cb = tqdm(
            total=directory_size, desc="Downloading...", unit='B', unit_scale=True
        ).update
        self._api.storage.download_directory(self._team_id, remote_path, local_path, progress_cb=progress_cb)

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
                            new_path = os.path.splitext(os.path.normpath(file_path))[0]
                            unpack_archive(file_path, new_path)
                            archives.append(file_path)
                            new_paths_to_scan.append(new_path)
                        except Exception as e:
                            logger.error(f"Error while unpacking '{file}': {repr(e)}")

            for archive in archives:
                silent_remove(archive)

# @TODO:
# [ ] - add timer
# [ ] - detect remote data format
# [ ] - windows junk if endswith Zone.Identifier?
# [x] - check if archive and unpack
# [x] - LAS format
# [ ] - preserve meta infos
# [ ] - merge meta.json if more than one
# [ ] - prepare test data for each modality and different formats (corner cases)
# [ ] - # ? add pointcloud shape
# [ ] - pcd extensions {'.ply', '.las', '.laz', '.xyz', '.pts', '.pcd'} ?
