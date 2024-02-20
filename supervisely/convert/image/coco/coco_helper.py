from copy import deepcopy
from typing import List

import cv2
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO

from supervisely import (
    Annotation,
    Bitmap,
    Label,
    ObjClass,
    PointLocation,
    Polygon,
    ProjectMeta,
    Rectangle,
    Tag,
    TagMeta,
    TagValueType,
    logger,
)
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.color import generate_rgb


# COCO Convert funcs
def create_supervisely_annotation(
    item: ImageConverter.Item, meta: ProjectMeta, coco_categories: List[dict]
):
    labels = []
    imag_tags = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    for object in item.ann_data:
        category_id = object.get("category_id")
        if category_id is None:
            continue
        obj_class_name = name_cat_id_map.get(category_id)
        if obj_class_name is None:
            logger.warn(f"Category with id {category_id} not found in categories list")
            continue
        segm = object.get("segmentation")
        curr_labels = []
        if segm is not None and len(segm) > 0:
            obj_class_polygon = meta.get_obj_class(obj_class_name)

            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    label = Label(figure, obj_class_polygon)
                    curr_labels.append(label)
            elif type(segm) is list and object["segmentation"]:
                figures = convert_polygon_vertices(object, item.shape)
                curr_labels.extend([Label(figure, obj_class_polygon) for figure in figures])
        labels.extend(curr_labels)
        bbox = object.get("bbox")
        if bbox is not None and len(bbox) == 4:
            if not obj_class_name.endswith("_bbox"):
                obj_class_name = add_tail(obj_class_name, "bbox")
            obj_class_rectangle = meta.get_obj_class(obj_class_name)
            if len(curr_labels) > 1:
                for label in curr_labels:
                    bbox = label.geometry.to_bbox()
                    labels.append(Label(bbox, obj_class_rectangle))
            else:
                x, y, w, h = bbox
                rectangle = Label(Rectangle(y, x, y + h, x + w), obj_class_rectangle)
                labels.append(rectangle)
        caption = object.get("caption")
        if caption is not None:
            imag_tags.append(Tag(meta.get_tag_meta("caption"), caption))
    return Annotation(item.shape, labels=labels, img_tags=imag_tags)


def convert_rle_mask_to_polygon(coco_ann):
    if type(coco_ann["segmentation"]["counts"]) is str:
        coco_ann["segmentation"]["counts"] = bytes(
            coco_ann["segmentation"]["counts"], encoding="utf-8"
        )
        mask = mask_util.decode(coco_ann["segmentation"])
    else:
        rle_obj = mask_util.frPyObjects(
            coco_ann["segmentation"],
            coco_ann["segmentation"]["size"][0],
            coco_ann["segmentation"]["size"][1],
        )
        mask = mask_util.decode(rle_obj)
    mask = np.array(mask, dtype=bool)
    if not np.any(mask):
        return []
    return Bitmap(mask).to_contours()


def convert_polygon_vertices(coco_ann, image_size):
    polygons = coco_ann["segmentation"]
    if all(type(coord) is float for coord in polygons):
        polygons = [polygons]
    if any(type(coord) is str for polygon in polygons for coord in polygon):
        return []
    exteriors = []
    for polygon in polygons:
        polygon = [polygon[i * 2 : (i + 1) * 2] for i in range((len(polygon) + 2 - 1) // 2)]
        exterior_points = [(width, height) for width, height in polygon]
        if len(exterior_points) == 0:
            continue
        exteriors.append(exterior_points)
    interiors = {idx: [] for idx in range(len(exteriors))}
    id2del = []
    for idx, exterior in enumerate(exteriors):
        temp_img = np.zeros(image_size + (3,), dtype=np.uint8)
        geom = Polygon([PointLocation(y, x) for x, y in exterior])
        geom.draw_contour(temp_img, color=[255, 255, 255])
        im = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        for idy, exterior2 in enumerate(exteriors):
            if idx == idy or idy in id2del:
                continue
            points_inside = [
                cv2.pointPolygonTest(contours[0], (x, y), False) > 0 for x, y in exterior2
            ]
            if all(points_inside):
                interiors[idx].append(deepcopy(exteriors[idy]))
                id2del.append(idy)
    for j in sorted(id2del, reverse=True):
        del exteriors[j]
    figures = []
    for exterior, interior in zip(exteriors, interiors.values()):
        exterior = [PointLocation(y, x) for x, y in exterior]
        interior = [[PointLocation(y, x) for x, y in points] for points in interior]
        figures.append(Polygon(exterior, interior))
    return figures


def generate_meta_from_annotation(coco: COCO, meta: ProjectMeta = None):
    obj_classes = []
    tag_metas = []
    colors = []

    ann_types = get_ann_types(coco)
    categories = coco.loadCats(ids=coco.getCatIds())
    if meta is None:
        meta = ProjectMeta()
    for category in categories:
        if category["name"] in [obj_class.name for obj_class in meta.obj_classes]:
            continue
        new_color = generate_rgb(colors)
        colors.append(new_color)
        if ann_types is not None:  # add skeleton support
            if "segmentation" in ann_types:
                obj_classes.append(ObjClass(category["name"], Polygon, new_color))
            if "bbox" in ann_types:
                obj_classes.append(
                    ObjClass(add_tail(category["name"], "bbox"), Rectangle, new_color)
                )
        for obj_class in obj_classes:
            existing_classes = [obj_class.name for obj_class in meta.obj_classes]
            if obj_class.name not in existing_classes:
                meta = meta.add_obj_class(obj_class)
        if ann_types is not None and "caption" in ann_types:
            tag_metas.append(TagMeta("caption", TagValueType.ANY_STRING))
        for tag_meta in tag_metas:
            existing_tags = [tag_meta.name for tag_meta in meta.tag_metas]
            if tag_meta.name not in existing_tags:
                meta = meta.add_tag_meta(tag_meta)
    return meta


def get_ann_types(coco: COCO) -> List[str]:
    ann_types = []
    annotation_ids = coco.getAnnIds()
    if any("bbox" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("bbox")
    if any("segmentation" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("segmentation")
    if any("caption" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("caption")
    return ann_types


def add_tail(body: str, tail: str):
    if " " in body:
        return f"{body} {tail}"
    return f"{body}_{tail}"


def coco_category_to_class_name(coco_categories):
    return {category["id"]: category["name"] for category in coco_categories}
