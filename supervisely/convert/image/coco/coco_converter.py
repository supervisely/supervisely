from typing import List

from pycocotools.coco import COCO

from supervisely import (
    Annotation,
    ObjClass,
    Polygon,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
)
from supervisely.convert.base_converter import AvailableImageFormats, BaseConverter
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import get_file_ext, list_files_recursively

COCO_ANN_KEYS = ["images", "annotations", "categories"]

class COCOConverter(BaseConverter):
    def __init__(self, input_data):
        
        self.classes = None
        self.tags = None
        self.meta = None
        super().__init__(input_data)

    def __str__(self):
        return AvailableImageFormats.COCO

    # @staticmethod
    # def validate_ann_format(ann_path):
    #     coco = COCO(ann_path)  # dont throw error if not COCO
    #     if not all(key in coco.dataset for key in COCO_ANN_KEYS):
    #         return False
    #     return True
    
    # def get_meta(self):
    #     return super().get_meta()
    
    # def get_items(self):
    #     return super().get_items()

    # def to_supervisely(self, image_path: str, ann_path: str):
    #     raise NotImplementedError()

    @property
    def ann_ext(self):
        return None # ? ".json" 
    
    @property
    def key_file_ext(self):
        return ".json"

    def require_key_file(self):
        return True
    
    def validate_ann_files(self):
        pass

    def validate_key_files(self):
        jsons = list_files_recursively(self.items, valid_extensions=[".json"])
        for key_file in jsons:
            coco = COCO(key_file)  # dont throw error if not COCO
            if not all(key in coco.dataset for key in COCO_ANN_KEYS):
                return False
            
            # [ ] @TODO: ADD KEYPOINTS SUPPORT
            def get_ann_types(coco: COCO) -> List[str]:
                ann_types = []
                annotation_ids = coco.getAnnIds()
                if any("bbox" in coco.anns[ann_id] for ann_id in annotation_ids):
                    ann_types.append("bbox")
                if any("segmentation" in coco.anns[ann_id] for ann_id in annotation_ids):
                    ann_types.append("segmentation")
                if any("caption" in coco.anns[ann_id] for ann_id in annotation_ids):
                    ann_types.append("caption")
                return ann_types
            
            def add_tail(body: str, tail: str):
                if " " in body:
                    return f"{body} {tail}"
                return f"{body}_{tail}"
            
            colors = []
            tag_metas = []
            ann_types = get_ann_types(coco)
            categories = coco.loadCats(ids=coco.getCatIds())
            for category in categories:
                if category["name"] in [obj_class.name for obj_class in g.META.obj_classes]:
                    continue
                new_color = generate_rgb(colors)
                colors.append(new_color)
            
            obj_classes = []
            if ann_types is not None:
                if "segmentation" in ann_types:
                    obj_classes.append(ObjClass(category["name"], Polygon, new_color))
                if "bbox" in ann_types:
                    obj_classes.append(
                        ObjClass(add_tail(category["name"], "bbox"), Rectangle, new_color)
                    )
                
            if self.meta is None:
                self.meta = ProjectMeta()
            self.meta = self.meta.add_obj_classes(obj_classes)
            if ann_types is not None and "caption" in ann_types:
                tag_metas.append(TagMeta("caption", TagValueType.ANY_STRING))
            self.meta = self.meta.add_tag_metas(tag_metas)           
        return True

    def get_meta(self):
        return self.meta

    def get_items(self): # -> generator?
        raise NotImplementedError()
    
    def to_supervisely(self, item_path: str, ann_path: str) -> Annotation:
        """Convert to Supervisely format."""

        self.meta = self.get_meta()
        raise NotImplementedError()