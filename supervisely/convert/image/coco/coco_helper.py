import os
import sys
import uuid
from copy import deepcopy
from typing import List

import cv2
import numpy as np


class HiddenCocoPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


from supervisely import (
    Annotation,
    Bitmap,
    GraphNodes,
    Label,
    Node,
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
from supervisely.geometry.graph import KeypointsTemplate
from supervisely.imaging.color import generate_rgb

conflict_classes = []


# COCO Convert funcs
def create_supervisely_annotation(
    item: ImageConverter.Item,
    meta: ProjectMeta,
    coco_categories: List[dict],
    renamed_classes: dict = None,
    renamed_tags: dict = None,
):
    labels = []
    imag_tags = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    renamed_classes = {} if renamed_classes is None else renamed_classes
    renamed_tags = {} if renamed_tags is None else renamed_tags
    for object in item.ann_data:
        caption = object.get("caption")
        if caption is not None:
            tag_name = renamed_tags.get("caption", "caption")
            imag_tags.append(Tag(meta.get_tag_meta(tag_name), caption))
        category_id = object.get("category_id")
        if category_id is None:
            continue
        obj_class_name = name_cat_id_map.get(category_id)
        if obj_class_name is None:
            logger.warn(f"Category with id {category_id} not found in categories list")
            continue
        renamed_class_name = renamed_classes.get(obj_class_name, obj_class_name)
        key = None
        segm = object.get("segmentation")
        curr_labels = []
        if segm is not None and len(segm) > 0:
            obj_class_polygon = meta.get_obj_class(renamed_class_name)
            if obj_class_polygon.geometry_type != Polygon:
                if obj_class_name not in conflict_classes:
                    geometry_name = obj_class_polygon.geometry_type.geometry_name().capitalize()
                    conflict_classes.append(obj_class_name)
                    logger.warn(
                        "Conflict in class geometry type: "
                        f"object class '{obj_class_name}' (category ID: {category_id}) "
                        f"has type '{geometry_name}', but expected type is 'Polygon'."
                    )
                continue
            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    key = uuid.uuid4().hex
                    label = Label(figure, obj_class_polygon, binding_key=key)
                    curr_labels.append(label)
            elif type(segm) is list and object["segmentation"]:
                figures = convert_polygon_vertices(object, item.shape)
                for figure in figures:
                    key = uuid.uuid4().hex
                    label = Label(figure, obj_class_polygon, binding_key=key)
                    curr_labels.append(label)

        keypoints = object.get("keypoints")
        if keypoints is not None:
            obj_class_keypoints = meta.get_obj_class(renamed_class_name)
            keypoints = list(get_coords(object["keypoints"]))
            coco_categorie, keypoint_names = None, None
            for cat in coco_categories:
                if cat["id"] == category_id and cat["supercategory"] == obj_class_name:
                    coco_categorie = cat
                    break
            if coco_categorie is not None:
                keypoint_names = coco_categorie.get("keypoints")
            if keypoint_names is not None:
                nodes = []
                for coords, keypoint_name in zip(keypoints, keypoint_names):
                    col, row, visibility = coords
                    if visibility in [0, 1]:
                        continue  # skip invisible keypoints

                    node = Node(label=keypoint_name, row=row, col=col)  # , disabled=v)
                    nodes.append(node)
                if len(nodes) != 0:
                    key = uuid.uuid4().hex
                    label = Label(GraphNodes(nodes), obj_class_keypoints, binding_key=key)
                    curr_labels.append(label)
        labels.extend(curr_labels)
        bbox = object.get("bbox")
        if bbox is not None and len(bbox) == 4:
            if not obj_class_name.endswith("bbox"):
                obj_class_name = add_tail(obj_class_name, "bbox")
            renamed_class_name = renamed_classes.get(obj_class_name, obj_class_name)
            obj_class_rectangle = meta.get_obj_class(renamed_class_name)
            if obj_class_rectangle.geometry_type != Rectangle:
                if obj_class_name not in conflict_classes:
                    geometry_name = obj_class_rectangle.geometry_type.geometry_name().capitalize()
                    conflict_classes.append(obj_class_name)
                    logger.warn(
                        "Conflict in class geometry type: "
                        f"object class '{obj_class_name}' (category ID: {category_id}) "
                        f"has type '{geometry_name}', but expected type is 'Rectangle'."
                    )
                continue
            if len(curr_labels) > 1:
                for label in curr_labels:
                    bbox = label.geometry.to_bbox()
                    labels.append(Label(bbox, obj_class_rectangle, binding_key=label.binding_key))
            else:
                if len(curr_labels) == 1:
                    key = curr_labels[0].binding_key
                x, y, w, h = bbox
                rectangle = Label(
                    Rectangle(y, x, y + h, x + w), obj_class_rectangle, binding_key=key
                )
                labels.append(rectangle)
    return Annotation(item.shape, labels=labels, img_tags=imag_tags)


def convert_rle_mask_to_polygon(coco_ann):
    import pycocotools.mask as mask_util  # pylint: disable=import-error

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


def generate_meta_from_annotation(coco, meta: ProjectMeta = None):
    from pycocotools.coco import COCO  # pylint: disable=import-error

    coco: COCO

    colors = []
    ann_types = get_ann_types(coco)
    if len(coco.cats) == 0:
        categories = []
    else:
        categories = coco.loadCats(ids=coco.getCatIds())
    if meta is None:
        meta = ProjectMeta()
    for category in categories:
        if category["name"] in [obj_class.name for obj_class in meta.obj_classes]:
            continue
        obj_classes = []
        new_color = generate_rgb(colors)
        colors.append(new_color)
        if ann_types is not None:  # add skeleton support
            if "keypoints" in ann_types and "keypoints" in category:
                geometry_config = None
                cat_num_kp = None
                cat_labels = category.get("keypoints", None)
                cat_edges = category.get("skeleton", None)
                if cat_labels is None:
                    cat_num_kp = []
                    for img_id, img_info in coco.imgs.items():
                        ann_p = coco.anns[img_id]
                        for ann in ann_p:
                            if ann["category_id"] == category["id"]:
                                cat_num_kp.append(ann["num_keypoints"])
                    cat_num_kp = max(cat_num_kp)

                geometry_config = create_custom_geometry_config(
                    num_keypoints=cat_num_kp, cat_labels=cat_labels, cat_edges=cat_edges
                )
                obj_class = ObjClass(
                    name=category["name"],
                    geometry_type=GraphNodes,
                    color=new_color,
                    geometry_config=geometry_config,
                )
                obj_classes.append(obj_class)
            elif "segmentation" in ann_types:
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
        tag_meta = TagMeta("caption", TagValueType.ANY_STRING)
        if meta.get_tag_meta(tag_meta.name) is None:
            meta = meta.add_tag_meta(tag_meta)
    return meta


def get_ann_types(coco) -> List[str]:
    from pycocotools.coco import COCO  # pylint: disable=import-error

    coco: COCO

    ann_types = []
    annotation_ids = coco.getAnnIds()
    if any("bbox" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("bbox")
    if any("segmentation" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("segmentation")
    if any("caption" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("caption")
    if any("keypoints" in coco.anns[ann_id] for ann_id in annotation_ids):
        ann_types.append("keypoints")
    return ann_types


def add_tail(body: str, tail: str):
    if " " in body:
        return f"{body} {tail}"
    return f"{body}_{tail}"


def coco_category_to_class_name(coco_categories):
    return {category["id"]: category["name"] for category in coco_categories}


def get_coords(keypoints):
    for i in range(0, len(keypoints), 3):
        yield keypoints[i : i + 3]


def create_custom_geometry_config(num_keypoints=None, cat_labels=None, cat_edges=None):
    template = KeypointsTemplate()

    if cat_labels is None:
        if num_keypoints is None:
            raise ValueError(
                "Number of keypoints can not be specified, please check your annotation (categories: num_keypoints)"
            )
        for p in list(range(num_keypoints)):
            template.add_point(label=str(p), row=0, col=0)
    else:
        for label in cat_labels:
            template.add_point(label=label, row=0, col=0)

    if cat_edges is not None and cat_labels is not None:
        for edge in cat_edges:
            template.add_edge(src=cat_labels[edge[0] - 1], dst=cat_labels[edge[1] - 1])
    else:
        logger.warn("Edges can not be mapped without skeleton, please check your annotation")
    return template
