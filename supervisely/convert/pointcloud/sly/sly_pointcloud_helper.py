from typing import List

from supervisely import (
    AnyGeometry,
    Cuboid,
    ObjClass,
    ProjectMeta,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["figures", "objects", "tags"]


def get_meta_from_annotation(ann_path: str, meta: ProjectMeta) -> ProjectMeta:
    ann_json = load_json_file(ann_path)
    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]

    if not all(key in ann_json for key in SLY_ANN_KEYS):
        logger.warn(
            f"Pointcloud Annotation file {ann_path} is not in Supervisely format"
        )
        return meta

    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(object["tags"], meta)
        meta = create_classes_from_annotation(object, meta)
    meta = create_tags_from_annotation(ann_json["tags"], meta)
    return meta


def create_tags_from_annotation(tags: List[dict], meta: ProjectMeta) -> ProjectMeta:
    for tag in tags:
        tag_name = tag["name"]
        tag_value = tag["value"]
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


def create_classes_from_annotation(object: dict, meta: ProjectMeta) -> ProjectMeta:
    class_name = object["classTitle"]
    geometry_type = object["geometryType"]
    if geometry_type == Cuboid.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Cuboid)
        existing_class = meta.get_obj_class(class_name)
        if existing_class is None:
            meta = meta.add_obj_class(obj_class)
        else:
            if existing_class.geometry_type != obj_class.geometry_type:
                obj_class = ObjClass(name=class_name, geometry_type=AnyGeometry)
                meta = meta.delete_obj_class(class_name)
                meta = meta.add_obj_class(obj_class)
    return meta
