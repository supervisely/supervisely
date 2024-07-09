from typing import List

from supervisely.geometry.helpers import GET_GEOMETRY_FROM_STR
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.graph import GraphNodes, KeypointsTemplate, NODES
from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.tag import TagJsonFields
from supervisely import ObjClass, ProjectMeta, TagMeta, TagValueType, logger

SLY_IMAGE_ANN_KEYS = ["objects", "tags", "size"]
SLY_OBJECT_KEYS = [LabelJsonFields.OBJ_CLASS_NAME, LabelJsonFields.TAGS, "geometryType"]  #, LabelJsonFields.GEOMETRY_TYPE] TODO: add geometry type
SLY_TAG_KEYS = [TagJsonFields.TAG_NAME, TagJsonFields.VALUE]

# Check the annotation format documentation at
def get_meta_from_annotation(ann_json: dict, meta: ProjectMeta) -> ProjectMeta:
    """Generate sly.ProjectMeta from JSON annotation file."""

    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]

    if not all(key in ann_json for key in SLY_IMAGE_ANN_KEYS):
        logger.warn(
            f"Some keys are missing in the annotation file. "
            "Check the annotation format documentation at: "
            "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/05_supervisely_format_images"
        )
        return meta

    for object in ann_json["objects"]:
        meta = create_tags_from_annotation(object["tags"], meta)
        meta = create_classes_from_annotation(object, meta)
    meta = create_tags_from_annotation(ann_json["tags"], meta)
    return meta


def create_tags_from_annotation(tags: List[dict], meta: ProjectMeta) -> ProjectMeta:
    for tag in tags:
        if not all(key in tag for key in SLY_TAG_KEYS):
            logger.warn(
                f"Tag in annotation file is not in Supervisely format. "
                "Read more about the Supervisely JSON format of tags in the documentation at: "
                "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/03_supervisely_format_tags"
            )
            continue
        tag_name = tag[TagJsonFields.TAG_NAME]
        tag_value = tag[TagJsonFields.VALUE]
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
    if not all(key in object for key in SLY_OBJECT_KEYS):
        logger.warn(
            f"Object in annotation file is not in Supervisely format: {object}. "
            "Read more about the Supervisely JSON format of objects in the documentation at: "
            "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/04_supervisely_format_objects"
        )
        return meta
    class_name = object["classTitle"]
    geometry_type_str = object["geometryType"]
    obj_class = None

    try:
        geometry_type = GET_GEOMETRY_FROM_STR(geometry_type_str)
    except KeyError:
        logger.warn(f"Unknown geometry type {geometry_type_str} for class {class_name}")
        return meta

    obj_class = None
    geometry_config = None
    if issubclass(geometry_type, GraphNodes):
        if NODES in object:
            geometry_config = KeypointsTemplate()
            for uuid, node in object[NODES].items():
                if "loc" in node and len(node["loc"]) == 2:
                    geometry_config.add_point(label=uuid, row=node["loc"][0], col=node["loc"][1])
    obj_class = ObjClass(name=class_name, geometry_type=geometry_type, geometry_config=geometry_config)
    existing_class = meta.get_obj_class(class_name)

    if obj_class is None:
        logger.warn(f"Failed to create object class for {class_name} with geometry type {geometry_type_str}")
        return meta

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

def annotation_high_level_validator(ann_json: dict) -> bool:
    """Check if annotation is probably in Supervisely format."""

    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if not all(key in ann_json for key in SLY_IMAGE_ANN_KEYS):
        return False
    for obj in ann_json["objects"]:
        if not all(key in obj for key in SLY_OBJECT_KEYS):
            return False
        for tag in obj["tags"]:
            if not all(key in tag for key in SLY_TAG_KEYS):
                return False
    for tag in ann_json["tags"]:
        if not all(key in tag for key in SLY_TAG_KEYS):
            return False
    return True

def get_image_size_from_annotation(ann_json: dict) -> tuple:
    """Get image size from annotation."""

    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if "size" not in ann_json:
        return None
    size = ann_json["size"]
    if "height" not in size or "width" not in size:
        return None
    return size["height"], size["width"]
