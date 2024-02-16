import os

from supervisely import (
    Annotation,
    Bitmap,
    GraphNodes,
    ObjClass,
    Point,
    PointLocation,
    Polygon,
    Polyline,
    ProjectMeta,
    Rectangle,
    TagMeta,
)
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.geometry.geometry import Geometry
from supervisely.io.fs import file_exists, get_file_ext, list_files_recursively
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["imageName", "imageId", "createdAt", "updatedAt", "annotation"]


# match items and anns on init?


class SLYImageConverter(ImageConverter):

    def __init__(self, input_data, items, annotations):
        self._input_data = input_data
        self._items = items
        self._annotations = annotations
        self._meta = None

    def __str__(self):
        return AvailableImageConverters.SLY

    @property
    def ann_ext(self):
        return ".json"

    def validate_ann_file(self, ann_path):
        if self._meta is None:
            if file_exists(ann_path):
                ann_json = load_json_file(ann_path)
                if all(key in ann_json for key in SLY_ANN_KEYS):
                    return True
            return False
        else:
            try:
                ann = Annotation.from_json(load_json_file(ann_path), self._meta)
                return True
            except Exception:
                return False

    def require_key_file(self):
        return True

    def validate_key_files(self):
        jsons = list_files_recursively(self._input_data, valid_extensions=[".json"])
        # TODO: find meta.json first
        for key_file in jsons:
            try:
                self._meta = ProjectMeta.from_json(load_json_file(key_file))
                return True
            except Exception:
                continue
        return False

    def get_meta(self):
        if self._meta is not None:
            return self._meta
        else:
            return self._generate_meta_from_anns()

    def _generate_meta_from_anns(self):
        meta = ProjectMeta()
        for ann_path in self._annotations:
            if self.validate_ann_file(ann_path):
                ann_json = load_json_file(ann_path)
                for object in ann_json["annotation"]["objects"]:
                    class_name = object["classTitle"]
                    geometry_type = object["geometryType"]

                    # @TODO: add better check for geometry type, add
                    if geometry_type == Bitmap.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=Bitmap)
                    elif geometry_type == Rectangle.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=Rectangle)
                    elif geometry_type == Point.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=Point)
                    elif geometry_type == Polygon.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=Polygon)
                    elif geometry_type == Polyline.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=Polyline)
                    elif geometry_type == PointLocation.geometry_name():
                        obj_class = ObjClass(name=class_name, geometry_type=PointLocation)
                    # elif geometry_type == GraphNodes.geometry_name():
                    #     geometry_config = None
                    #     obj_class = ObjClass(name=class_name, geometry_type=GraphNodes)
                    existing_class = meta.get_obj_class(class_name)
                    if existing_class is None:
                        meta = meta.add_obj_class(obj_class)
                    else:
                        continue

                # [ ] @TODO: add tags
            else:
                continue
        self._meta = meta
        return self._meta

    def get_items(self):
        return self._items

    def to_supervisely(self, item_path: str, ann_path: str) -> Annotation:
        """Convert to Supervisely format."""

        if self._meta is None:
            self._meta = self.get_meta()
        raise NotImplementedError()
