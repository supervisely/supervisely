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
    def __init__(self, input_data, items, annotations):
        
        self.input_data = input_data
        self.items = items
        self.annotations = []
        
        self.classes = None
        self.tags = None
        self.meta = None
        super().__init__(self.input_data, self.items, self.annotations)

    def __str__(self):
        return AvailableImageFormats.COCO

    @property
    def ann_ext(self):
        return None # ? ".json" 
    
    @property
    def key_file_ext(self):
        return ".json"

    def require_key_file(self):
        return True
    
    def validate_ann_file(self, ann_path: str):
        if self.meta is not None:
            return True
        return False

    def validate_key_files(self):
        jsons = list_files_recursively(self.input_data, valid_extensions=[".json"])
        for key_file in jsons:
            coco = COCO(key_file)  # wont throw error if not COCO
            if not all(key in coco.dataset for key in COCO_ANN_KEYS):
                continue
            
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
            
            if self.meta is None:
                self.meta = ProjectMeta()
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
                        ObjClass(add_tail(category["name"], "bbox"), Rectangle, new_color)
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
                    
            self.annotations.append(key_file)
            
            
        if self.meta is None:
            return False
        return True

    def get_meta(self):
        return self.meta

    def get_items(self): # -> generator?
        return self.items.keys()
    
    def get_anns(self):
        return self.annotations
    
    def to_supervisely(self, item_path: str, ann_path: str) -> Annotation:
        """Convert to Supervisely format."""

        self.meta = self.get_meta()
        raise NotImplementedError()