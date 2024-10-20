from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_location import row_col_list_to_points
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.convert.image.image_helper import validate_image_bounds

POSSIBLE_SHAPE_TYPES = ["polygonlabels", "rectanglelabels", "brushlabels"]
POSSIBLE_TAGS_TYPES = ["choices"]


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """get bit string from bytes data"""
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def convert_rle_mask_to_sly(coords: List[int], h: int, w: int) -> Optional[Bitmap]:
    """Convert Label Studio RLE mask to Supervisely Bitmap."""

    rle_input = InputStream(bytes2bit(coords))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    mask = np.reshape(out, [h, w, 4])[:, :, 3].astype(np.bool)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return Bitmap(mask)


def convert_polygon_to_sly(coords: List[List[int]], h: int, w: int) -> Polygon:
    """Convert Label Studio polygon coordinates to Supervisely Polygon object."""

    coords = [[int(row * h / 100), int(col * w / 100)] for (col, row) in coords]
    points = row_col_list_to_points(coords, do_round=True)
    return Polygon(points)


def convert_rectangle_to_sly(coords: List[int], h: int, w: int) -> Rectangle:
    """Convert Label Studio rectangle coordinates to Supervisely Rectangle object."""

    x, y, rect_w, rect_h = coords
    r, b = x + rect_w, y + rect_h
    l, t, r, b = int(x * w / 100), int(y * h / 100), int(r * w / 100), int(b * h / 100)
    return Rectangle(t, l, b, r)


convert_geom_func_map = {
    "polygonlabels": (convert_polygon_to_sly, ("points",)),
    "rectanglelabels": (convert_rectangle_to_sly, ("x", "y", "width", "height")),
    "brushlabels": (convert_rle_mask_to_sly, ("rle",)),
}


def convert_geometry_to_sly(
    shape_type: str, data: Dict[str, Any], h: int, w: int
) -> Optional[Geometry]:
    try:
        convert_func, keys = convert_geom_func_map.get(shape_type, (None, None))
        if convert_func is None:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        coords = [data.get(key) for key in keys] if len(keys) > 1 else data.get(keys[0])
        geometry = convert_func(coords, h, w)
        return geometry
    except Exception as e:
        logger.warn(f"Failed to convert geometry: {shape_type}. Reason: {repr(e)}")
        return None


def generate_new_name_for_meta(
    meta: ProjectMeta,
    name: str,
    expected_type: type,
    i: Optional[int] = None,
    is_tag: bool = False,
) -> str:
    """Generate new name for tag or class in project meta."""

    new_name = f"{name}_{i}" if i else name
    item = meta.get_tag_meta(new_name) if is_tag else meta.get_obj_class(new_name)
    if item is None:
        return new_name
    if is_tag and item.value_type == expected_type:
        return new_name
    if not is_tag and item.geometry_type == expected_type:
        return new_name
    return generate_new_name_for_meta(meta, name, expected_type, i + 1 if i else 1, is_tag)


def _get_or_create_tag_or_cls(meta: ProjectMeta, name: str, geometry_name: Optional[str] = None):
    """Private function to get or create tag or class in project meta."""

    is_tag = geometry_name is None
    item_type = TagValueType.NONE if is_tag else GET_GEOMETRY_FROM_STR(geometry_name)
    item = meta.get_tag_meta(name) if is_tag else meta.get_obj_class(name)
    if item is None:
        item = TagMeta(name, item_type) if is_tag else ObjClass(name, item_type)
        meta = meta.add_tag_meta(item) if is_tag else meta.add_obj_class(item)
        return item, meta
    if is_tag and item.value_type != item_type or not is_tag and item.geometry_type != item_type:
        new_name = generate_new_name_for_meta(meta, name, item_type, is_tag=is_tag)
        if new_name != name:
            logger.warn(
                f"Type mismatch for {'tag' if is_tag else 'class'} '{name}'. Renamed to '{new_name}'"
            )
            item = TagMeta(new_name, item_type) if is_tag else ObjClass(new_name, item_type)
            meta = meta.add_tag_meta(item) if is_tag else meta.add_obj_class(item)
    return item, meta


def get_or_create_tag_meta(meta: ProjectMeta, tag_name: str) -> Tuple[TagMeta, ProjectMeta]:
    """Get or create TagMeta in project meta."""

    return _get_or_create_tag_or_cls(meta, tag_name)


def get_or_create_obj_cls(
    meta: ProjectMeta, cls_name: str, geometry_name: str
) -> Tuple[ObjClass, ProjectMeta]:
    """Get or create ObjClass in project meta."""

    return _get_or_create_tag_or_cls(meta, cls_name, geometry_name)


def create_supervisely_annotation(image_path: str, ann: Dict, meta: ProjectMeta) -> Annotation:
    """Create Supervisely annotation from Label Studio annotation."""

    sly_ann = Annotation.from_img_path(image_path)
    h, w = sly_ann.img_size

    relations = []  # list of relations (from_id, to_id)
    key_label_map = defaultdict(list)  # map for object id to created list of Label objects
    img_tags = []  # list of tags

    objects = ann.get("result", [])
    for obj in objects:
        item_type = obj.get("type")
        if item_type == "relation":
            relations.append((obj["from_id"], obj["to_id"]))
            continue
        item_id = obj.get("id")
        item_data = obj.get("value", {})
        item_name = item_data.get(item_type, [])
        if len(item_name) == 0:
            continue
        item_name = item_name[0]
        if item_type in POSSIBLE_SHAPE_TYPES:
            geom = convert_geometry_to_sly(item_type, item_data, h, w)
            if geom is None:
                continue
            obj_cls, meta = get_or_create_obj_cls(meta, item_name, geom.geometry_name())
            key_label_map[item_id].append(Label(geom, obj_cls))
        elif item_type in POSSIBLE_TAGS_TYPES:
            tag_meta, meta = get_or_create_tag_meta(meta, item_name)
            img_tags.append(Tag(tag_meta))

    res_labels = []
    for from_id, to_id in relations:
        key = uuid4().hex
        labels = key_label_map.pop(from_id, []) + key_label_map.pop(to_id, [])
        res_labels.extend([label.clone(binding_key=key) for label in labels])

    for labels in key_label_map.values():
        res_labels.extend(labels)

    res_labels = validate_image_bounds(res_labels, Rectangle.from_size(sly_ann.img_size))
    sly_ann = sly_ann.clone(labels=res_labels, img_tags=img_tags)
    return sly_ann, meta
