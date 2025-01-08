from typing import Dict, List, Union

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
from supervisely.io.fs import get_file_name
from supervisely.io.json import load_json_file

SLY_ANN_KEYS = ["figures", "objects", "tags"]


def get_meta_from_annotation(ann_path: str, meta: Union[ProjectMeta, None]) -> ProjectMeta:
    if meta is None:
        meta = ProjectMeta()
    ann_json = load_json_file(ann_path)
    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]

    if not all(key in ann_json for key in SLY_ANN_KEYS):
        logger.warn(f"Pointcloud Annotation file {ann_path} is not in Supervisely format")
        return meta

    object_key_to_name = {}
    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(object["tags"], meta)
        object_key_to_name[object["key"]] = object["classTitle"]
    for fig in ann_json["figures"]:
        meta = create_classes_from_annotation(fig, meta, object_key_to_name)
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
    figure: dict, meta: ProjectMeta, object_key_to_name: dict
) -> ProjectMeta:
    obj_key = figure.get("objectKey")
    if obj_key is None:
        return meta
    class_name = object_key_to_name[obj_key]
    geometry_type = figure["geometryType"]
    obj_class = None
    if geometry_type == Cuboid.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Cuboid3d)
    elif geometry_type == Cuboid3d.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Cuboid3d)

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


def find_related_items(
    name: str,
    possible_img_ext: List[str],
    rimg_dict: Dict[str, str],
    rimg_ann_dict: Dict[str, str],
):
    name_noext = get_file_name(name)
    for ext in possible_img_ext:
        for img_name in [f"{name_noext}{ext}", f"{name}{ext}"]:
            rimg_path = rimg_dict.get(img_name)
            if rimg_path:
                for ann_path in [f"{name_noext}.json", f"{name}.json", f"{img_name}.json"]:
                    rimg_ann_path = rimg_ann_dict.get(ann_path)
                    if rimg_ann_path:
                        return rimg_path, rimg_ann_path
    return None, None
