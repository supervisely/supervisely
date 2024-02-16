from pycocotools.coco import COCO

import supervisely.convert.image.coco.coco_helper as coco_helper
from supervisely import (
    Annotation,
    ObjClass,
    Polygon,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import list_files_recursively

COCO_ANN_KEYS = ["images", "annotations", "categories"]


class COCOConverter(ImageConverter):
    def __init__(self, input_data, items, annotations):
        self._input_data = input_data
        self._items = items
        self._annotations = annotations
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.COCO

    @property
    def ann_ext(self):
        return None  # ? ".json"

    @property
    def key_file_ext(self):
        return ".json"

    def require_key_file(self):
        return True

    def validate_ann_file(self, ann_path: str):
        if self._meta is not None:
            return True
        return False

    def validate_key_files(self):
        jsons = list_files_recursively(self._input_data, valid_extensions=[".json"])
        for key_file in jsons:
            coco = COCO(key_file)  # wont throw error if not COCO
            if not all(key in coco.dataset for key in COCO_ANN_KEYS):
                continue

            colors = []
            tag_metas = []
            ann_types = coco_helper.get_ann_types(coco)
            categories = coco.loadCats(ids=coco.getCatIds())

            if self._meta is None:
                self._meta = ProjectMeta()
            for category in categories:
                if category["name"] in [obj_class.name for obj_class in self.meta.obj_classes]:
                    continue
                new_color = generate_rgb(colors)
                colors.append(new_color)

                obj_classes = []
                if ann_types is not None:
                    if "segmentation" in ann_types:
                        obj_classes.append(ObjClass(category["name"], Polygon, new_color))
                    if "bbox" in ann_types:
                        obj_classes.append(
                            ObjClass(
                                coco_helper.add_tail(category["name"], "bbox"), Rectangle, new_color
                            )
                        )

                for obj_class in obj_classes:
                    existing_classes = [obj_class.name for obj_class in self.meta.obj_classes]
                    if obj_class.name not in existing_classes:
                        self.meta = self.meta.add_obj_class(obj_class)

                if ann_types is not None and "caption" in ann_types:
                    tag_metas.append(TagMeta("caption", TagValueType.ANY_STRING))

                for tag_meta in tag_metas:
                    existing_tags = [tag_meta.name for tag_meta in self.meta.tag_metas]
                    if tag_meta.name not in existing_tags:
                        self.meta = self.meta.add_tag_meta(tag_meta)

            coco_anns = coco.imgToAnns
            coco_images = coco.imgs
            coco_items = coco_images.items()

            for img_id, img_info in coco_items:
                img_ann = coco_anns[img_id]
                img_shape = (img_info["height"], img_info["width"])
                for item in self.items:
                    filename = img_info["file_name"]
                    if filename == item.name:
                        item.update(item.path, img_ann, img_shape, {"categories": categories})

        if self._meta is None:
            return False
        return True

    def get_meta(self):
        return self._meta

    def get_items(self):  # -> generator?
        return self._items

    def to_supervisely(self, item: ImageConverter.Item, meta: ProjectMeta) -> Annotation:
        """Convert to Supervisely format."""
        if item.ann_data is None:
            if item.shape is not None:
                return Annotation(item.shape)
            else:
                return Annotation.from_img_path(item.path)

        ann = coco_helper.create_supervisely_ann(
            meta, item.custom_data["categories"], item.ann_data, item.shape
        )
        return ann
