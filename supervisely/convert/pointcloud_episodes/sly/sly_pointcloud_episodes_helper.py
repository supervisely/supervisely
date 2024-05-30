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
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["frames", "objects", "tags", "framesCount"]


def get_meta_from_annotation(ann_path: str, meta: ProjectMeta) -> ProjectMeta:
    ann_json = load_json_file(ann_path)
    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]

    if not all(key in ann_json for key in SLY_ANN_KEYS):
        logger.warn(
            f"Pointcloud Episode Annotation file {ann_path} is not in Supervisely format"
        )
        return meta

    object_key_to_name = {}
    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(object["tags"], meta)
        object_key_to_name[object["key"]] = object["classTitle"]
    for frame in ann_json["frames"]:
        meta = create_classes_from_annotation(frame, meta, object_key_to_name)
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


def create_classes_from_annotation(
        frame: list, meta: ProjectMeta, object_key_to_name: dict
) -> ProjectMeta:
    for figure in frame["figures"]:
        obj_key = figure.get("objectKey")
        if obj_key is None:
            continue
        class_name = object_key_to_name[obj_key]
        geometry_type = figure["geometryType"]
        obj_class = None
        if geometry_type == Cuboid.geometry_name():
            obj_class = ObjClass(name=class_name, geometry_type=Cuboid3d)
        elif geometry_type == Cuboid3d.geometry_name():
            obj_class = ObjClass(name=class_name, geometry_type=Cuboid3d)
        elif geometry_type == Pointcloud.geometry_name():
            obj_class = ObjClass(name=class_name, geometry_type=Pointcloud)

        if obj_class is not None:
            existing_class = meta.get_obj_class(class_name)
            if existing_class is None:
                meta = meta.add_obj_class(obj_class)
            else:
                if existing_class.geometry_type != obj_class.geometry_type:
                    meta = meta.delete_obj_class(class_name)
                    obj_class = ObjClass(name=class_name, geometry_type=AnyGeometry)
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