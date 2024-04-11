from typing import List

from supervisely import (
    AnyGeometry,
    Bitmap,
    Mask3D,
    ObjClass,
    Polygon,
    Polyline,
    ProjectMeta,
    Rectangle,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.io.json import load_json_file

SLY_VOLUME_ANN_KEYS = ["volumeMeta", "planes", "spatialFigures", "planes"]


def get_meta_from_annotation(ann_path: str, meta: ProjectMeta) -> ProjectMeta:
    ann_json = load_json_file(ann_path)
    if not all(key in ann_json for key in SLY_VOLUME_ANN_KEYS):
        logger.warn(f"Volume Annotation file {ann_path} is not in Supervisely format")
        return meta

    objects_geom_map = match_objects_to_geometries(ann_json)
    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(object["tags"], meta)
        meta = create_classes_from_annotation(
            object, objects_geom_map.get(object["classTitle"]), meta
        )
    meta = create_tags_from_annotation(ann_json["tags"], meta)
    return meta


def match_objects_to_geometries(ann_json: dict) -> dict:
    objects = ann_json["objects"]
    spatial_figures = ann_json["spatialFigures"]
    planes = ann_json["planes"]

    matched_objects = {}
    for obj in objects:
        obj_class_name = obj["classTitle"]

        if obj_class_name in matched_objects:
            continue

        obj_key = obj.get("key")
        obj_id = obj.get("id")
        if obj_id is None and obj_key is None:
            logger.warn(
                f"Couldn't generate meta for object class: {obj['classTitle']}. Object has no key or id"
            )
            continue

        for spatial_figure in spatial_figures:
            spatial_figure_obj_id = spatial_figure.get("objectId")
            spatial_figure_obj_key = spatial_figure.get("objectKey")

            if spatial_figure_obj_id is not None and obj_id is not None:
                if spatial_figure["objectId"] == obj_id:
                    matched_objects[obj_class_name] = spatial_figure["geometryType"]
                    break
            elif spatial_figure_obj_key is not None and obj_key is not None:
                if spatial_figure["objectKey"] == obj_key:
                    matched_objects[obj_class_name] = spatial_figure["geometryType"]
                    break

        for plane in planes:
            slices = plane["slices"]
            for slice in slices:
                figures = slice["figures"]
                for figure in figures:
                    figure_obj_id = figure.get("objectId")
                    figure_obj_key = figure.get("objectKey")
                    if obj_id is not None and figure_obj_id is not None:
                        if obj_id == figure_obj_id:
                            matched_objects[obj_class_name] = figure["geometryType"]
                            break
                    elif obj_key is not None and figure_obj_key is not None:
                        if obj_key == figure_obj_key:
                            matched_objects[obj_class_name] = figure["geometryType"]
                            break
    return matched_objects


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


def create_classes_from_annotation(object: dict, object_geometry, meta: ProjectMeta) -> ProjectMeta:
    class_name = object["classTitle"]
    geometry_type = object_geometry
    if geometry_type is None:  # should not happen
        return meta
    if geometry_type == Mask3D.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Mask3D)
    elif geometry_type == Rectangle.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Rectangle)
    elif geometry_type == Polygon.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Polygon)
    elif geometry_type == Bitmap.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=Bitmap)
    elif geometry_type == AnyGeometry.geometry_name():
        obj_class = ObjClass(name=class_name, geometry_type=AnyGeometry)
    elif geometry_type == Polyline:
        obj_class = ObjClass(name=class_name, geometry_type=Polyline)

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
