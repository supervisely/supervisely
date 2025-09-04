from typing import List

from supervisely import ObjClass, ProjectMeta, TagMeta, TagValueType, logger
from supervisely.annotation.annotation import AnnotationJsonFields
from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.tag import TagJsonFields
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.constants import LOC
from supervisely.geometry.graph import NODES, GraphNodes, KeypointsTemplate
from supervisely.geometry.helpers import GET_GEOMETRY_FROM_STR

SLY_IMAGE_ANN_KEYS = [
    AnnotationJsonFields.LABELS,
    AnnotationJsonFields.IMG_TAGS,
    AnnotationJsonFields.IMG_SIZE,
]
SLY_OBJECT_KEYS = [
    LabelJsonFields.OBJ_CLASS_NAME,
    LabelJsonFields.TAGS,
    LabelJsonFields.GEOMETRY_TYPE,
]
SLY_TAG_KEYS = [
    TagJsonFields.TAG_NAME,
    # TagJsonFields.VALUE
]


# Check the annotation format documentation at
def get_meta_from_annotation(ann_json: dict, meta: ProjectMeta) -> ProjectMeta:
    """Generate sly.ProjectMeta from JSON annotation file."""

    if "annotation" in ann_json:
        ann_json = ann_json.get("annotation", {})

    if not all(key in ann_json for key in SLY_IMAGE_ANN_KEYS):
        logger.warning(
            f"Some keys are missing in the annotation file. "
            "Check the annotation format documentation at: "
            "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/05_supervisely_format_images"
        )
        return meta

    ann_objects = ann_json.get(AnnotationJsonFields.LABELS, [])
    for object in ann_objects:
        obj_tags = object.get(LabelJsonFields.TAGS, None)
        if obj_tags is None:
            logger.warning(
                f"Key '{LabelJsonFields.TAGS}' for object tags is missing in the annotation file. Tags will not be added to the meta."
            )
            obj_tags = []
        meta = create_tags_from_annotation(obj_tags, meta)
        meta = create_classes_from_annotation(object, meta)
    img_tags = ann_json.get(AnnotationJsonFields.IMG_TAGS, None)
    if img_tags is None:
        logger.warning(
            f"Key '{AnnotationJsonFields.IMG_TAGS}' for image tags is missing in the annotation file. Tags will not be added to the meta."
        )
        img_tags = []
    meta = create_tags_from_annotation(img_tags, meta)
    return meta


def create_tags_from_annotation(tags: List[dict], meta: ProjectMeta) -> ProjectMeta:
    for tag in tags:
        if not all(key in tag for key in SLY_TAG_KEYS):
            logger.warning(
                f"Tag in annotation file is not in Supervisely format. "
                "Read more about the Supervisely JSON format of tags in the documentation at: "
                "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/03_supervisely_format_tags"
            )
            continue
        tag_name = tag[TagJsonFields.TAG_NAME]
        tag_value = tag.get(TagJsonFields.VALUE)
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
        logger.warning(
            f"Object in annotation file is not in Supervisely format: {object}. "
            "Read more about the Supervisely JSON format of objects in the documentation at: "
            "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/04_supervisely_format_objects"
        )
        return meta
    class_name = object[LabelJsonFields.OBJ_CLASS_NAME]
    geometry_type_str = object[LabelJsonFields.GEOMETRY_TYPE]
    obj_class = None

    try:
        geometry_type = GET_GEOMETRY_FROM_STR(geometry_type_str)
    except KeyError:
        logger.warning(f"Unknown geometry type {geometry_type_str} for class {class_name}")
        return meta

    obj_class = None
    geometry_config = None
    if issubclass(geometry_type, GraphNodes):
        if NODES in object:
            geometry_config = KeypointsTemplate()
            for uuid, node in object[NODES].items():
                if LOC in node and len(node[LOC]) == 2:
                    geometry_config.add_point(label=uuid, row=node[LOC][0], col=node[LOC][1])
    obj_class = ObjClass(
        name=class_name, geometry_type=geometry_type, geometry_config=geometry_config
    )
    existing_class = meta.get_obj_class(class_name)

    if obj_class is None:
        logger.warning(
            f"Failed to create object class for {class_name} with geometry type {geometry_type_str}"
        )
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
        for obj in ann_json[AnnotationJsonFields.LABELS]:
            obj[LabelJsonFields.OBJ_CLASS_NAME] = renamed_classes.get(
                obj[LabelJsonFields.OBJ_CLASS_NAME], obj[LabelJsonFields.OBJ_CLASS_NAME]
            )
            for tag in obj[LabelJsonFields.TAGS]:
                tag[TagJsonFields.TAG_NAME] = renamed_tags.get(
                    tag[TagJsonFields.TAG_NAME], tag[TagJsonFields.TAG_NAME]
                )
    if renamed_tags:
        for tag in ann_json[AnnotationJsonFields.IMG_TAGS]:
            tag[TagJsonFields.TAG_NAME] = renamed_tags.get(
                tag[TagJsonFields.TAG_NAME], tag[TagJsonFields.TAG_NAME]
            )
    return ann_json


def annotation_high_level_validator(ann_json: dict) -> bool:
    """Check if annotation is probably in Supervisely format."""

    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if not all(key in ann_json for key in SLY_IMAGE_ANN_KEYS):
        return False
    for obj in ann_json[AnnotationJsonFields.LABELS]:
        if not all(key in obj for key in SLY_OBJECT_KEYS):
            return False
        for tag in obj[LabelJsonFields.TAGS]:
            if not all(key in tag for key in SLY_TAG_KEYS):
                return False
    for tag in ann_json[AnnotationJsonFields.IMG_TAGS]:
        if not all(key in tag for key in SLY_TAG_KEYS):
            return False
    return True


def get_image_size_from_annotation(ann_json: dict) -> tuple:
    """Get image size from annotation."""

    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if "size" not in ann_json:
        return None
    size = ann_json[AnnotationJsonFields.IMG_SIZE]
    if (
        AnnotationJsonFields.IMG_SIZE_HEIGHT not in size
        or AnnotationJsonFields.IMG_SIZE_WIDTH not in size
    ):
        return None
    return size[AnnotationJsonFields.IMG_SIZE_HEIGHT], size[AnnotationJsonFields.IMG_SIZE_WIDTH]
