from typing import List

from supervisely import (
    AnyGeometry,
    Bitmap,
    ObjClass,
    Point,
    Polygon,
    Polyline,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["size", "framesCount", "frames", "objects", "tags"]


def get_meta_from_annotation(ann_path: str, meta: ProjectMeta) -> ProjectMeta:
    ann_json = load_json_file(ann_path)
    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if not all(key in ann_json for key in SLY_ANN_KEYS):
        logger.warn(f"VideoAnnotation file {ann_path} is not in Supervisely format")
        return meta

    object_key_to_name = {}
    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(meta, object["tags"])
        object_key_to_name[object["key"]] = object["classTitle"]
    for frame in ann_json["frames"]:
        meta = create_classes_from_annotation(frame, meta, object_key_to_name)
    meta = create_tags_from_annotation(meta, ann_json["tags"])
    return meta


def create_tags_from_annotation(meta: ProjectMeta, tags: List[dict]) -> ProjectMeta:
    for tag in tags:
        tag_name = tag["name"]
        tag_value = tag.get("value")
        if tag_value is None:
            tag_meta = TagMeta(tag_name, TagValueType.NONE)
        elif isinstance(tag_value, int) or isinstance(tag_value, float):
            tag_meta = TagMeta(tag_name, TagValueType.ANY_NUMBER)
        else:
            tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)

        # check existing tag_meta in meta
        existing_tag = meta.get_tag_meta(tag_name)
        if existing_tag is None:
            meta = meta.add_tag_meta(tag_meta)
    return meta


def create_classes_from_annotation(
    frame: dict, meta: ProjectMeta, object_key_to_name: dict
) -> ProjectMeta:
    for fig in frame["figures"]:
        obj_key = fig.get("objectKey")
        if obj_key is None:
            continue
        class_name = object_key_to_name[obj_key]
        geometry_type = fig["geometryType"]
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

        # @TODO: add better check for geometry type, add
        # elif geometry_type == GraphNodes.geometry_name():
        #     geometry_config = None
        #     obj_class = ObjClass(name=class_name, geometry_type=GraphNodes)
        existing_class = meta.get_obj_class(class_name)
        if existing_class is None:
            meta = meta.add_obj_class(obj_class)
        else:
            if existing_class.geometry_type != obj_class.geometry_type:
                obj_class = ObjClass(name=class_name, geometry_type=AnyGeometry)
                meta = meta.delete_obj_class(class_name)
                meta = meta.add_obj_class(obj_class)
    return meta

def rename_in_json(ann_json, renamed_classes=None, renamed_tags=None):
    if renamed_classes:
        for obj in ann_json["objects"]:
            obj["classTitle"] = renamed_classes.get(obj["classTitle"], obj["classTitle"])
            for tag in obj["tags"]:
                tag["name"] = renamed_tags.get(tag["name"], tag["name"])
    if renamed_tags:
        for tag in ann_json["tags"]:
            tag["name"] = renamed_tags.get(tag["name"], tag["name"])
    return ann_json