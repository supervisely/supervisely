import os
import shutil
import sys
import uuid
from copy import deepcopy
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np

from supervisely._utils import generate_free_name
from supervisely.api.image_api import ImageApi
from supervisely.imaging.image import read as sly_image
from supervisely.io.fs import get_file_name_with_ext
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.task.progress import tqdm_sly

COCO_INSTANCES_FILE = "coco_instances.json"
COCO_CAPTIONS_FILE = "coco_captions.json"


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
from supervisely.convert.image.image_helper import validate_image_bounds
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
    for coco_data in item.ann_data:
        caption = coco_data.get("caption")
        if caption is not None:
            tag_name = renamed_tags.get("caption", "caption")
            imag_tags.append(Tag(meta.get_tag_meta(tag_name), caption))
        category_id = coco_data.get("category_id")
        if category_id is None:
            continue
        obj_class_name = name_cat_id_map.get(category_id)
        if obj_class_name is None:
            logger.warning(f"Category with id {category_id} not found in categories list")
            continue
        renamed_class_name = renamed_classes.get(obj_class_name, obj_class_name)
        key = None

        segm = coco_data.get("segmentation")
        keypoints = coco_data.get("keypoints")
        bbox = coco_data.get("bbox")

        if len([f for f in (segm, keypoints, bbox) if f]) > 1:
            # create a binding key if more than one of the following fields are present
            key = uuid.uuid4().hex

        curr_labels = []
        if segm is not None and len(segm) > 0:
            obj_class_polygon = meta.get_obj_class(renamed_class_name)
            if obj_class_polygon.geometry_type != Polygon:
                if obj_class_name not in conflict_classes:
                    geometry_name = obj_class_polygon.geometry_type.geometry_name().capitalize()
                    conflict_classes.append(obj_class_name)
                    logger.warning(
                        "Conflict in class geometry type: "
                        f"object class '{obj_class_name}' (category ID: {category_id}) "
                        f"has type '{geometry_name}', but expected type is 'Polygon'."
                    )
                continue
            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(coco_data)
                for polygon in polygons:
                    curr_labels.append(Label(polygon, obj_class_polygon, binding_key=key))
            elif type(segm) is list and coco_data["segmentation"]:
                polygons = convert_polygon_vertices(coco_data, item.shape)
                for polygon in polygons:
                    curr_labels.append(Label(polygon, obj_class_polygon, binding_key=key))

        if keypoints is not None:
            obj_class_keypoints = meta.get_obj_class(renamed_class_name)
            keypoints = list(get_coords(keypoints))
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
                    label = Label(GraphNodes(nodes), obj_class_keypoints, binding_key=key)
                    curr_labels.append(label)
        labels.extend(curr_labels)

        if bbox is not None and len(bbox) == 4:
            if not obj_class_name.endswith("bbox"):
                obj_class_name = add_tail(obj_class_name, "bbox")
            renamed_class_name = renamed_classes.get(obj_class_name, obj_class_name)
            obj_class_rectangle = meta.get_obj_class(renamed_class_name)
            if obj_class_rectangle.geometry_type != Rectangle:
                if obj_class_name not in conflict_classes:
                    geometry_name = obj_class_rectangle.geometry_type.geometry_name().capitalize()
                    conflict_classes.append(obj_class_name)
                    logger.warning(
                        "Conflict in class geometry type: "
                        f"object class '{obj_class_name}' (category ID: {category_id}) "
                        f"has type '{geometry_name}', but expected type is 'Rectangle'."
                    )
                continue
            x, y, w, h = bbox
            geometry = Rectangle(y, x, y + h, x + w)
            labels.append(Label(geometry, obj_class_rectangle, binding_key=key))
    labels = validate_image_bounds(labels, Rectangle.from_size(item.shape))
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


def convert_polygon_vertices(coco_ann, image_size: Tuple[int, int]):
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
        logger.warning("Edges can not be mapped without skeleton, please check your annotation")
    return template


def _get_graph_info(idx, obj_class):
    data = {"supercategory": obj_class.name, "id": idx, "name": obj_class.name}
    kp = {i: n["label"] for i, n in obj_class.geometry_config["nodes"].items()}
    keys = {k: j for j, k in enumerate(list(kp.keys()), 1)}
    edges = obj_class.geometry_config["edges"]
    sk = [[keys[e["src"]], keys[e["dst"]]] for e in edges]
    data["keypoints"] = list(kp.values())
    data["skeleton"] = sk
    return data


def get_categories_from_meta(meta: ProjectMeta) -> List[Dict[str, Any]]:
    """Get categories from Supervisely project meta."""
    cat = lambda idx, c: {"supercategory": c.name, "id": idx, "name": c.name}
    return [
        cat(idx, c) if c.geometry_type != GraphNodes else _get_graph_info(idx, c)
        for idx, c in enumerate(meta.obj_classes, start=1)
    ]


def extend_mask_up_to_image(
    binary_mask: np.ndarray, image_shape: Tuple[int, int], origin: PointLocation
) -> np.ndarray:
    """Extend binary mask up to image shape."""
    y, x = origin.col, origin.row
    new_mask = np.zeros(image_shape, dtype=binary_mask.dtype)
    try:
        new_mask[x : x + binary_mask.shape[0], y : y + binary_mask.shape[1]] = binary_mask
    except ValueError as e:
        raise ValueError(
            f"Binary mask size {binary_mask.shape} with origin {origin} "
            f"exceeds image boundaries {image_shape}"
        ) from e
    return new_mask


def coco_segmentation_rle(segmentation: np.ndarray) -> Dict[str, Any]:
    """Convert COCO segmentation to RLE format."""
    binary_mask = np.asfortranarray(segmentation)
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def sly_ann_to_coco(
    ann: Annotation,
    coco_image_id: int,
    class_mapping: Dict[str, int],
    coco_ann: Optional[Union[Dict, List]] = None,
    last_label_id: Optional[int] = None,
    coco_captions: Optional[Union[Dict, List]] = None,
    last_caption_id: Optional[int] = None,
) -> Tuple[List, List]:
    """
    Convert Supervisely annotation to COCO format annotation ("annotations" field).

    :param coco_image_id: Image id in COCO format.
    :type coco_image_id: int
    :param class_mapping: Dictionary that maps class names to class ids.
    :type class_mapping: Dict[str, int]
    :param coco_ann: COCO annotation in dictionary or list format to append new annotations.
    :type coco_ann: Union[Dict, List], optional
    :param last_label_id: Last label id in COCO format to continue counting.
    :type last_label_id: int, optional
    :param coco_captions: COCO captions in dictionary or list format to append new captions.
    :type coco_captions: Union[Dict, List], optional
    :return: Tuple with list of COCO objects and list of COCO captions.
    :rtype: :class:`tuple`


    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.convert.image.coco.coco_helper import sly_ann_to_coco


        coco_instances = dict(
            info=dict(
                description="COCO dataset converted from Supervisely",
                url="None",
                version=str(1.0),
                year=2025,
                contributor="Supervisely",
                date_created="2025-01-01 00:00:00",
            ),
            licenses=[dict(url="None", id=0, name="None")],
            images=[],
            annotations=[],
            categories=get_categories_from_meta(meta),  # [{"supercategory": "lemon", "id": 1, "name": "lemon"}, ...]
        )

        ann = sly.Annotation.from_json(ann_json, meta)
        image_id = 11
        label_id = 222
        class_mapping = {obj_cls.name: idx for idx, obj_cls in enumerate(meta.obj_classes)}

        curr_coco_ann, _ = sly_ann_to_coco(ann, image_id, class_mapping, coco_instances, label_id)
        # or
        # curr_coco_ann, _ = sly_ann_to_coco(ann, image_id, class_mapping, label_id=label_id)
        # coco_instances["annotations"].extend(curr_coco_ann)

        label_id += len(curr_coco_ann)
        image_id += 1
    """

    coco_obj_template = lambda x, y, z: {
        "id": x,
        "segmentation": [],
        "area": 0,
        "iscrowd": 0,
        "image_id": y,
        "bbox": [],
        "category_id": z,
    }
    label_id = last_label_id + 1 if last_label_id is not None else 1
    if isinstance(coco_ann, dict):
        coco_ann = coco_ann["annotations"]
    if isinstance(coco_ann, list):
        label_id = len(coco_ann) + 1

    last_caption_id = last_caption_id + 1 if last_caption_id is not None else 1
    if isinstance(coco_captions, dict):
        coco_captions = coco_captions["annotations"]
    if isinstance(coco_captions, list):
        last_caption_id = len(coco_captions) + 1

    def _update_inst_results(label_id, coco_ann, coco_obj, res):
        label_id += 1
        if isinstance(coco_ann, list):
            coco_ann.append(coco_obj)
        res.append(coco_obj)
        return label_id

    def _get_common_bbox(labels, sly_bbox=False, approx=False):
        bboxes = [l.geometry.to_bbox() for l in labels]
        x = min([bbox.left for bbox in bboxes])
        y = min([bbox.top for bbox in bboxes])
        max_x = max([bbox.right for bbox in bboxes])
        max_y = max([bbox.bottom for bbox in bboxes])
        if approx:
            x, y, max_x, max_y = x - 10, y - 10, max_x + 10, max_y + 10
        if sly_bbox:
            return Rectangle(top=y, left=x, bottom=max_y, right=max_x)
        return [x, y, max_x - x, max_y - y]

    def _create_keypoints_obj(label, cat_id, label_id, coco_image_id):
        nodes_dict = label.obj_class.geometry_config["nodes"]
        keypoint_uuid_labels = {i: d["label"] for i, d in nodes_dict.items()}
        keypoints = []
        for key in keypoint_uuid_labels.keys():
            if key not in label.geometry.nodes:
                keypoints.extend([0, 0, 0])
            else:
                loc = label.geometry.nodes[key].location
                keypoints.extend([loc.col, loc.row, 2])
        coco_obj = coco_obj_template(label_id, coco_image_id, cat_id)
        coco_obj["keypoints"] = keypoints
        coco_obj["num_keypoints"] = len(keypoint_uuid_labels)
        x, y = keypoints[0::3], keypoints[1::3]
        x0, x1, y0, y1 = (np.min(x), np.max(x), np.min(y), np.max(y))
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        coco_obj["area"] = int((x1 - x0) * (y1 - y0))
        coco_obj["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        return coco_obj

    def _update_caption_results(caption_id, coco_captions, caption, res):
        caption_id += 1
        if isinstance(coco_captions, list):
            coco_captions.append(caption)
        res.append(caption)
        return caption_id

    res_inst = []  # result list of COCO objects

    h, w = ann.img_size
    for binding_key, labels in ann.get_bindings().items():
        if binding_key is None:
            polygons = [l for l in labels if l.geometry.name() == Polygon.geometry_name()]
            masks = [l for l in labels if l.geometry.name() == Bitmap.geometry_name()]
            bboxes = [l for l in labels if l.geometry.name() == Rectangle.geometry_name()]
            graphs = [l for l in labels if l.geometry.name() == GraphNodes.geometry_name()]
            # polygons = [l for l in labels if l.obj_class.geometry_type == Polygon]
            for label in polygons + bboxes + masks:
                cat_id = class_mapping[label.obj_class.name]
                coco_obj = coco_obj_template(label_id, coco_image_id, cat_id)
                coco_obj["bbox"] = _get_common_bbox([label])
                coco_obj["area"] = label.geometry.area
                if label.geometry.name() == Polygon.geometry_name():
                    poly = label.geometry.to_json()["points"]["exterior"]
                    poly = np.array(poly).flatten().astype(float).tolist()
                    coco_obj["segmentation"] = [poly]
                elif label.geometry.name() == Bitmap.geometry_name():
                    segmentation = extend_mask_up_to_image(
                        label.geometry.data, (h, w), label.geometry.origin
                    )
                    coco_obj["segmentation"] = coco_segmentation_rle(segmentation)

                label_id = _update_inst_results(label_id, coco_ann, coco_obj, res_inst)

            for label in graphs:
                cat_id = class_mapping[label.obj_class.name]
                new_obj = _create_keypoints_obj(label, cat_id, label_id, coco_image_id)
                label_id = _update_inst_results(label_id, coco_ann, new_obj, res_inst)

            continue

        bboxes = [l for l in labels if l.geometry.name() == Rectangle.geometry_name()]
        polygons = [l for l in labels if l.geometry.name() == Polygon.geometry_name()]
        masks = [l for l in labels if l.geometry.name() == Bitmap.geometry_name()]
        graphs = [l for l in labels if l.geometry.name() == GraphNodes.geometry_name()]

        need_to_process_separately = len(masks) > 0 and len(polygons) > 0
        bbox_matched_w_mask = False
        bbox_matched_w_poly = False

        if len(graphs) > 0:
            if len(masks) > 0 or len(polygons) > 0:
                logger.warning(
                    "Keypoints and Polygons/Bitmaps in one binding key are not supported. "
                    "Objects will be converted separately."
                )
            if len(graphs) > 1:
                logger.warning(
                    "Multiple Keypoints in one binding key are not supported. "
                    "Only the first graph will be converted."
                )
            cat_id = class_mapping[graphs[0].obj_class.name]
            coco_obj = _create_keypoints_obj(graphs[0], cat_id, label_id, coco_image_id)
            label_id = _update_inst_results(label_id, coco_ann, coco_obj, res_inst)

        # convert Bitmap to Polygon
        if len(masks) > 0:
            for label in masks:
                cat_id = class_mapping[label.obj_class.name]
                coco_obj = coco_obj_template(label_id, coco_image_id, cat_id)
                segmentation = extend_mask_up_to_image(
                    label.geometry.data, (h, w), label.geometry.origin
                )
                coco_obj["segmentation"] = coco_segmentation_rle(segmentation)
                coco_obj["area"] = label.geometry.area
                if len(bboxes) > 0 and not need_to_process_separately:
                    found = _get_common_bbox(bboxes, sly_bbox=True, approx=True)
                    new = _get_common_bbox([label], sly_bbox=True)
                    bbox_matched_w_mask = found.contains(new)
                coco_obj["bbox"] = _get_common_bbox(bboxes if bbox_matched_w_mask else [label])
                label_id = _update_inst_results(label_id, coco_ann, coco_obj, res_inst)

        # process polygons
        if len(polygons) > 0:
            cat_id = class_mapping[polygons[0].obj_class.name]
            coco_obj = coco_obj_template(label_id, coco_image_id, cat_id)
            if len(bboxes) > 0 and not need_to_process_separately:
                found = _get_common_bbox(bboxes, sly_bbox=True, approx=True)
                new = _get_common_bbox(polygons, sly_bbox=True)
                bbox_matched_w_poly = found.contains(new)

            polys = [l.geometry.to_json()["points"]["exterior"] for l in polygons]
            polys = [np.array(p).flatten().astype(float).tolist() for p in polys]
            coco_obj["segmentation"] = polys
            coco_obj["area"] = sum([l.geometry.area for l in polygons])
            coco_obj["bbox"] = _get_common_bbox(bboxes if bbox_matched_w_poly else polygons)
            label_id = _update_inst_results(label_id, coco_ann, coco_obj, res_inst)

        # process bboxes separately if they are not matched with masks/polygons
        if len(bboxes) > 0 and not bbox_matched_w_poly and not bbox_matched_w_mask:
            for label in bboxes:
                cat_id = class_mapping[label.obj_class.name]
                coco_obj = coco_obj_template(label_id, coco_image_id, cat_id)
                coco_obj["bbox"] = _get_common_bbox([label])
                coco_obj["area"] = label.geometry.area

                label_id = _update_inst_results(label_id, coco_ann, coco_obj, res_inst)

    is_caption = lambda t: t.meta.name == "caption" and t.meta.value_type == TagValueType.ANY_STRING
    caption_tags = [tag for tag in ann.img_tags if is_caption(tag)]

    res_captions = []  # result list of COCO captions
    for tag in caption_tags:
        caption = {
            "image_id": coco_image_id,
            "id": last_caption_id,
            "caption": tag.value,
        }
        last_caption_id = _update_caption_results(
            last_caption_id, coco_captions, caption, res_captions
        )

    return res_inst, res_captions


def has_caption_tag(meta: ProjectMeta) -> bool:
    tag = meta.get_tag_meta("caption")
    return tag is not None and tag.value_type == TagValueType.ANY_STRING


def create_coco_ann_template(meta: ProjectMeta) -> Dict[str, Any]:
    now = datetime.now()
    coco_ann = dict(
        info=dict(
            description="COCO dataset converted from Supervisely",
            url="None",
            version=str(1.0),
            year=now.year,
            contributor="Supervisely",
            date_created=now.strftime("%Y-%m-%d %H:%M:%S"),
        ),
        licenses=[dict(url="None", id=0, name="None")],
        images=[],
        annotations=[],
        categories=[],
    )
    coco_ann["categories"] = get_categories_from_meta(meta)
    return coco_ann


def sly_ds_to_coco(
    dataset: Dataset,
    meta: ProjectMeta,
    return_type: Literal["path", "dict"] = "path",
    dest_dir: Optional[str] = None,
    copy_images: bool = False,
    with_captions: bool = False,
    log_progress: bool = False,
    progress_cb: Optional[Callable] = None,
) -> Union[str, Tuple[str, str], Dict, Tuple[Dict, Dict]]:
    """
    Convert Supervisely dataset to COCO format.

    Note: Depending on the `return_type` and `with_captions` parameters, the function returns different values.

    :param dataset: Supervisely dataset.
    :type dataset: :class:`Dataset<supervisely.project.dataset.Dataset>`
    :param meta: Project meta information.
    :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`
    :param return_type: Type of return value.
                        If 'path', returns paths to COCO dataset files.
                        If 'dict', returns COCO dataset dictionaries.
    :param dest_dir: Destination path to save COCO dataset.
    :type dest_dir: :class:`str`, optional
    :param copy_images: If True, copies images to the destination directory.
    :type copy_images: :class:`bool`, optional
    :param with_captions: If True, returns COCO captions.
    :type with_captions: :class:`bool`, optional
    :param log_progress: If True, logs the progress of the conversion.
    :type log_progress: :class:`bool`, optional
    :param progress_cb: Callback function to track the progress of the conversion.
    :type progress_cb: :class:`Callable`, optional
    :return:
            If return_type is 'path', returns paths to COCO dataset file or file (instances or instances and captions).
            If return_type is 'dict', returns COCO dataset dictionary or dictionaries (instances or instances and captions).
    :rtype: :class:`tuple`

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.convert.image.coco.coco_helper import sly_ds_to_coco

        project_path = "/home/admin/work/supervisely/projects/lemons_annotated"
        project = sly.Project(project_path, sly.OpenMode.READ)

        for ds in project.datasets:
            dest_dir = "/home/admin/work/supervisely/projects/lemons_annotated/ds1"
            coco_json, coco_captions, coco_json_path, coco_captions_path = sly_ds_to_coco(ds, project.meta, save=True, dest_dir=dest_dir)
    """
    dest_dir = Path(dataset.path).parent / "coco" if dest_dir is None else Path(dest_dir)
    save_json = return_type == "path"
    if save_json is True:
        annotations_dir = dest_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
    if copy_images is True:
        images_dir = dest_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc=f"Converting dataset '{dataset.short_name}' to COCO format",
            total=len(dataset),
        ).update

    coco_ann = create_coco_ann_template(meta)
    coco_captions = create_coco_ann_template(meta) if with_captions else None

    image_coco = lambda info, idx: dict(
        license="None",
        file_name=info.name,
        url="None",
        height=info.height,
        width=info.width,
        date_captured=info.created_at,
        id=idx,
        sly_id=info.id,
    )

    class_mapping = {cls.name: idx for idx, cls in enumerate(meta.obj_classes, start=1)}
    label_id = 0
    caption_id = 0
    for image_idx, name in enumerate(dataset.get_items_names(), 1):
        img_path = dataset.get_img_path(name)
        img_name = get_file_name_with_ext(img_path)
        ann_path = dataset.get_ann_path(name)
        img_info_path = dataset.get_img_info_path(name)

        if copy_images:
            dst_img_path = images_dir / img_name
            shutil.copy(img_path, dst_img_path)

        ann = Annotation.load_json_file(ann_path, meta)
        if ann.img_size is None or ann.img_size == (0, 0) or ann.img_size == (None, None):
            img = sly_image(img_path)
            ann = ann.clone(img_size=[img.shape[0], img.shape[1]])

        if os.path.exists(img_info_path):
            image_info_json = load_json_file(img_info_path)
        else:
            now = datetime.now()
            image_info_json = {
                "id": None,
                "name": img_name,
                "height": ann.img_size[0],
                "width": ann.img_size[1],
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        image_info = ImageApi._convert_json_info(ImageApi(None), image_info_json)

        coco_ann["images"].append(image_coco(image_info, image_idx))
        if with_captions is True:
            # pylint: disable=unsubscriptable-object
            coco_captions["images"].append(image_coco(image_info, image_idx))
            # pylint: enable=unsubscriptable-object

        insts, captions = ann.to_coco(
            image_idx, class_mapping, coco_ann, label_id, coco_captions, caption_id
        )
        label_id += len(insts)
        caption_id += len(captions)

        if progress_cb is not None:
            progress_cb(1)

    ann_path = None
    captions_path = None
    if save_json is True:
        logger.info("Saving COCO annotations to disk...")
        ann_path = str(annotations_dir / COCO_INSTANCES_FILE)
        dump_json_file(coco_ann, ann_path)
        logger.info(f"Saved COCO instances to '{ann_path}'")

        if with_captions is True:
            captions_path = str(annotations_dir / COCO_CAPTIONS_FILE)
            dump_json_file(coco_captions, captions_path)
            logger.info(f"Saved COCO captions to '{captions_path}'")

            return ann_path, captions_path
        return ann_path
    if with_captions:
        return coco_ann, coco_captions
    return coco_ann


def sly_project_to_coco(
    project: Union[Project, str],
    dest_dir: Optional[str] = None,
    copy_images: bool = False,
    with_captions: bool = False,
    log_progress: bool = True,
    progress_cb: Optional[Callable] = None,
) -> None:
    """
    Convert Supervisely project to COCO format.

    :param project: Supervisely project.
    :type project: :class:`Project<supervisely.project.project.Project>` or :class:`str`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param copy_images: Copy images to destination directory.
    :type copy_images: :class:`bool`, optional
    :param with_captions: Return COCO captions.
    :type with_captions: :class:`bool`, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project).
    :type progress_cb: callable, optional
    :return: None
    :rtype: NoneType

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project
        project_directory = "/home/admin/work/supervisely/source/project"

        # Convert Project to COCO format
        sly.Project(project_directory).to_coco(log_progress=True)
    """
    if isinstance(project, str):
        project = Project(project, mode=OpenMode.READ)

    dest_dir = Path(dest_dir) if dest_dir is not None else Path(project.directory).parent / "coco"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc="Converting Supervisely project to COCO format", total=project.total_items
        ).update

    used_ds_names = set()
    for ds in project.datasets:
        ds: Dataset
        coco_dir = generate_free_name(used_ds_names, ds.short_name, extend_used_names=True)
        ds.to_coco(
            meta=project.meta,
            return_type="path",
            dest_dir=dest_dir / coco_dir,
            copy_images=copy_images,
            with_captions=with_captions,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
        logger.info(f"Dataset '{ds.short_name}' has been converted to COCO format.")
    logger.info(f"Project '{project.name}' has been converted to COCO format.")


def to_coco(
    input_data: Union[Project, Dataset, str],
    dest_dir: Optional[str] = None,
    meta: Optional[ProjectMeta] = None,
    copy_images: bool = True,
    with_captions: bool = False,
    log_progress: bool = True,
    progress_cb: Optional[Callable] = None,
) -> Union[None, str]:
    """
    Universal function to convert Supervisely project or dataset to COCO format.
    Note:
        - For better compatibility, please pass named arguments explicitly. Otherwise, the function may not work as expected.
            You can use the dedicated functions for each data type:

                - :func:`sly.convert.sly_project_to_coco()`
                - :func:`sly.convert.sly_ds_to_coco()`

        - If the input_data is a Project, the dest_dir parameters are required.
        - If the input_data is a Dataset, the meta and dest_dir parameters are required.

    :param input_data: Supervisely project, dataset, or path to the project or dataset.
    :type input_data: :class:`Project<supervisely.project.project.Project>`, :class:`Dataset<supervisely.project.dataset.Dataset>` or :class:`str`

    # Project or Dataset conversion arguments:
    :param dest_dir: Destination directory to save project or dataset in COCO format.
    :type dest_dir: :class:`str`, optional
    :param meta: Project meta information (required for dataset conversion).
    :type meta: :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`, optional
    :param copy_images: Copy images to destination directory
    :type copy_images: :class:`bool`, optional
    :param with_captions: If True, returns COCO captions
    :type with_captions: :class:`bool`, optional
    :param log_progress: Show uploading progress bar
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project or dataset).
    :type progress_cb: callable, optional

    :return: None
    :rtype: NoneType

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project in Supervisely format
        project_directory = "./source/project"
        project_fs = sly.Project(project_directory, sly.OpenMode.READ)

        # Convert Project to COCO format
        sly.convert.to_coco(project_directory, dest_dir="./coco")
        # or
        sly.convert.to_coco(project_fs, dest_dir="./coco")

        # Convert Dataset to COCO format
        # dataset: sly.Dataset
        sly.convert.to_coco(dataset, dest_dir="./coco", meta=project_fs.meta)
    """
    if isinstance(input_data, str):
        try:
            input_data = Project(input_data, mode=OpenMode.READ)
        except Exception:
            try:
                input_data = Dataset(input_data, mode=OpenMode.READ)
            except Exception:
                raise ValueError("Please check the path or the input data.")

    if isinstance(input_data, Project):
        return sly_project_to_coco(
            project=input_data,
            dest_dir=dest_dir,
            copy_images=copy_images,
            with_captions=with_captions,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    if isinstance(input_data, Dataset):
        if meta is None:
            raise ValueError("Project meta information is required for dataset conversion.")
        return sly_ds_to_coco(
            dataset=input_data,
            meta=meta,
            return_type="path",
            dest_dir=dest_dir,
            copy_images=copy_images,
            with_captions=with_captions,
            log_progress=log_progress,
            progress_cb=progress_cb,
        )
    raise ValueError("Unsupported input type. Only Project or Dataset are supported.")
