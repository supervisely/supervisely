import os

from supervisely import Api, ProjectType
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.video.video_converter import VideoConverter

# windows junk if endswith Zone.Identifier


# class Converter:
class ImportManager:
    def __init__(self, input_data, project_type: ProjectType):
        if not os.path.exists(input_data):
            raise RuntimeError(f"Directory does not exist: {input_data}")

        self._input_data = input_data
        self._items = []
        self._modality = project_type
        self._converter = self.get_converter()
        self._api = Api.from_env()

    @property
    def modality(self):
        return self._modality

    @property
    def converter(self):
        return self._converter

    def get_converter(self):
        """Return correct converter"""
        if self._modality == ProjectType.IMAGES.value:
            return ImageConverter(self._input_data)._converter
        elif self._modality == ProjectType.VIDEOS.value:
            return VideoConverter(self._input_data)._converter
        # elif self.modality == ProjectType.POINT_CLOUDS.value:
        #     return PointCloudConverter(input_data)
        # elif self.modality == ProjectType.VOLUMES.value:
        #     return VolumeConverter(input_data)

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        self.converter.upload_dataset(self._api, dataset_id)
