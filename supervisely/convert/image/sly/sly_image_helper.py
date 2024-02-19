from typing import Callable, List

from supervisely import (
    Bitmap,
    ObjClass,
    Point,
    PointLocation,
    Polygon,
    Polyline,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
)
from supervisely.io.json import load_json_file


def get_meta_from_annotations(annotations: List[str], validate_ann_func: Callable) -> ProjectMeta:
    meta = ProjectMeta()
    for ann_path in annotations:
        if validate_ann_func(ann_path):
            ann_json = load_json_file(ann_path)
            for object in ann_json["annotation"]["objects"]:
                meta = create_tags_from_annotation(meta, object["tags"])
                meta = create_classes_from_annotation(meta, object)
            meta = create_tags_from_annotation(meta, ann_json["annotation"]["tags"])
        else:
            continue
    return meta


def create_tags_from_annotation(meta: ProjectMeta, tags: List[dict]) -> ProjectMeta:
    for tag in tags:
        tag_name = tag["name"]
        tag_value = tag["value"]
        if tag_value is None:
            tag_meta = TagMeta(tag_name, TagValueType.NONE)
        else:
            tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)

        # check existing tag_meta in meta
        existing_tag = meta.get_tag_meta(tag_name)
        if existing_tag is None:
            meta = meta.add_tag_meta(tag_meta)
    return meta


def create_classes_from_annotation(meta: ProjectMeta, object: dict) -> ProjectMeta:
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
    return meta
