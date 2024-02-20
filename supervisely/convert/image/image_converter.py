import supervisely.imaging.image as image
from supervisely import Annotation, Api, ProjectMeta, batched, logger
from supervisely.convert.base_converter import BaseConverter


class ImageConverter(BaseConverter):
    class Item(BaseConverter.BaseItem):
        def __init__(self, item_path, ann_data=None, shape=None, custom_data={}):
            self._path = item_path
            self._ann_data = ann_data
            self._type = "image"
            if shape is None:
                img = image.read(item_path)
                self._shape = img.shape[:2]
            else:
                self._shape = shape
            self._custom_data = custom_data

        def create_empty_annotation(self):
            return Annotation(self._shape)

    def __init__(self, input_data):
        self._input_data = input_data
        self._meta = None
        self._items = []
        self._converter = self._detect_format()

    @property
    def format(self):
        return self._converter.format

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    def _detect_format(self):
        found_formats = []
        all_converters = ImageConverter.__subclasses__()
        for converter in all_converters:
            if converter.__name__ == "BaseConverter":
                continue
            converter = converter(self._input_data)
            if converter.validate_format():
                if len(found_formats) > 1:
                    raise RuntimeError(
                        f"Multiple formats detected: {found_formats}. "
                        "Mixed formats are not supported yet."
                    )
                found_formats.append(converter)

        if len(found_formats) == 0:
            logger.info(f"No valid dataset formats detected. Only image will be processed")
            return self  # TODO: list items if no valid format detected

        if len(found_formats) == 1:
            return found_formats[0]

    def upload_dataset(self, api: Api, dataset_id: int, batch_size: int = 50):
        """Upload converted data to Supervisely"""

        dataset = api.dataset.get_info_by_id(dataset_id)
        meta_json = api.project.get_meta(dataset.project_id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.merge(self._meta)

        api.project.update_meta(dataset.project_id, meta)

        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []
            for item in batch:
                ann = self.to_supervisely(item, meta)
                item_names.append(item.name)
                item_paths.append(item.path)
                anns.append(ann)

            img_infos = api.image.upload_paths(dataset_id, item_names, item_paths)
            img_ids = [img_info.id for img_info in img_infos]
            api.annotation.upload_anns(img_ids, anns)

        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")


# @TODO:
# COCO
# [ ] - Implement Skeleton support
# [ ] - Implement detailed coco label validation
# Supervisely
# [ ] - Implement keypoints support
# [ ] - Add ann keys validation to method `generate_meta_from_annotation()``
