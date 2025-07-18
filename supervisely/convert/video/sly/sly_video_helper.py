from typing import List

from supervisely import (
    AnyGeometry,
    Bitmap,
    GraphNodes,
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
from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.tag import TagJsonFields
from supervisely.geometry.graph import KeypointsTemplate
from supervisely.io.json import load_json_file
from supervisely.video_annotation.constants import (
    FIGURES,
    FRAMES,
    FRAMES_COUNT,
    IMG_SIZE,
    INDEX,
    KEY,
    OBJECT_KEY,
    OBJECTS,
    TAGS,
)

SLY_ANN_KEYS = [IMG_SIZE, FRAMES_COUNT, FRAMES, OBJECTS, TAGS]
SLY_VIDEO_OBJECT_KEYS = [LabelJsonFields.OBJ_CLASS_NAME, LabelJsonFields.TAGS, KEY]
SLY_TAG_KEYS = [
    TagJsonFields.TAG_NAME,
    # TagJsonFields.VALUE
]
SLY_FRAME_KEYS = [FIGURES, INDEX]
SLY_FIGURE_KEYS = [
    KEY,
    OBJECT_KEY,
    "geometryType",
]  # , LabelJsonFields.GEOMETRY_TYPE] TODO: add geometry type


def get_meta_from_annotation(ann_path: str, meta: ProjectMeta) -> ProjectMeta:
    ann_json = load_json_file(ann_path)
    if "annotation" in ann_json:
        ann_json = ann_json["annotation"]
    if not all(key in ann_json for key in SLY_ANN_KEYS):
        logger.warning(
            f"VideoAnnotation file {ann_path} is not in Supervisely format. "
            "Check the annotation format documentation at: "
            "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/06_supervisely_format_videos"
        )
        return meta

    object_key_to_name = {}
    for object in ann_json[OBJECTS]:
        if not all(key in object for key in SLY_VIDEO_OBJECT_KEYS):
            logger.warning(
                f"Object in annotation file is not in Supervisely format: {object}. "
                "Read more about the Supervisely JSON format of objects in the documentation at: "
                "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/06_supervisely_format_videos"
            )
            continue
        meta = create_tags_from_annotation(meta, object[TAGS])
        object_key_to_name[object[KEY]] = object[LabelJsonFields.OBJ_CLASS_NAME]
    for frame in ann_json[FRAMES]:
        if not all(key in frame for key in SLY_FRAME_KEYS):
            logger.warning(
                f"Frame in annotation file is not in Supervisely format: {frame}."
                "Read more about the Supervisely JSON format of frames in the documentation at: "
                "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/06_supervisely_format_videos"
            )
            continue
        meta = create_classes_from_annotation(frame, meta, object_key_to_name)
    meta = create_tags_from_annotation(meta, ann_json[TAGS])
    return meta


def create_tags_from_annotation(meta: ProjectMeta, tags: List[dict]) -> ProjectMeta:
    for tag in tags:
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


def create_classes_from_annotation(
    frame: dict, meta: ProjectMeta, object_key_to_name: dict
) -> ProjectMeta:
    for fig in frame[FIGURES]:
        if not all(key in fig for key in SLY_FIGURE_KEYS):
            logger.warning(
                f"Figure in annotation file is not in Supervisely format: {fig}. "
                "Read more about the Supervisely JSON format of figures in the documentation at: "
                "https://docs.supervisely.com/customization-and-integration/00_ann_format_navi/06_supervisely_format_videos"
            )
            continue
        obj_key = fig.get(OBJECT_KEY)
        if obj_key is None:
            continue
        class_name = object_key_to_name[obj_key]
        geometry_type = fig["geometryType"]
        obj_class = None
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
        elif geometry_type == GraphNodes.geometry_name():
            if not all(key in fig for key in ["geometry", "geometryType"]):
                continue
            geometry = fig["geometry"]
            if "nodes" not in geometry:
                continue
            template = KeypointsTemplate()
            for uuid, node in geometry["nodes"].items():
                if "loc" not in node or len(node["loc"]) != 2:
                    continue
                template.add_point(label=uuid, row=node["loc"][0], col=node["loc"][1])
            obj_class = ObjClass(
                name=class_name, geometry_type=GraphNodes, geometry_config=template
            )

        existing_class = meta.get_obj_class(class_name)
        if obj_class is None:
            logger.warning(f"Object class {class_name} is not in Supervisely format.")
            continue
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
        for obj in ann_json[OBJECTS]:
            obj_cls_name = obj[LabelJsonFields.OBJ_CLASS_NAME]
            obj[LabelJsonFields.OBJ_CLASS_NAME] = renamed_classes.get(obj_cls_name, obj_cls_name)
            for tag in obj[TAGS]:
                tag_name = tag[TagJsonFields.TAG_NAME]
                tag[TagJsonFields.TAG_NAME] = renamed_tags.get(tag_name, tag_name)
    if renamed_tags:
        for tag in ann_json[TAGS]:
            tag_name = tag[TagJsonFields.TAG_NAME]
            tag[TagJsonFields.TAG_NAME] = renamed_tags.get(tag_name, tag_name)
    return ann_json
