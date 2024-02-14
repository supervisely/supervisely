from typing import Literal

from supervisely import ProjectType
from supervisely.convert.base_format import BaseFormat
from supervisely.convert.image.image_converter import ImageFormatConverter
from supervisely import Api, batched


# class Converter:
class ImportManager:
    def __init__(
        self, input_data, save_path=None, output_type: Literal["folder", "archive"] = "folder"
    ):
        # input_data date - folder / archive / link / team files
        # if save_path is None - save to the same level folder
        self.input_data = input_data
        self.modality = self._detect_modality(input_data)
        self.converter = None
        self.api = Api.from_env()


    def _detect_modality(self, data):
        """Detect modality of input data (images, videos, pointclouds, volumes)"""
        raise NotImplementedError()
    
    def get_converter(self):
        """Return correct converter"""
        if self.modality == ProjectType.IMAGES:
            self.converter = ImageFormatConverter(self.input_data)
        elif self.modality == ProjectType.VIDEOS:
            # self.converter = VideoFormatConverter(input_data)
            pass
        return self.converter

    def upload_dataset(self, dataset_id):
        """Upload converted data to Supervisely"""
        meta = self.converter.get_meta()
        items = self.converter.get_items()
        for batch in batched(items, batch_size=50):
            item_names = list(batch.keys())
            img_paths = [item["image"] for item in batch]
            ann_paths = [item["ann"] for item in batch]

            anns = []
            for img_path, ann_path in zip(img_paths, ann_paths):
                ann = self.converter.to_supervisely(img_path, ann_path, meta)
                anns.append(ann)

            img_infos = self.api.image.upload_paths(dataset_id, item_names, img_paths)
            img_ids = [img_info.id for img_info in img_infos]
            self.api.annotation.upload_anns(img_ids, anns)

# Using:
# converter: BaseFormat = Converter("/path/to/folder")  # read and return correct converter

# converter.to_supervisely()
# converter.to_yolo()
