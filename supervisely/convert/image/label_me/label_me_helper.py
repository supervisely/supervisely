import base64
import io
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation, row_col_list_to_points
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.fs import file_exists
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.convert.image.image_helper import validate_image_bounds

labelme_shape_types_to_sly_map = {
    "polygon": Polygon,
    "rectangle": Rectangle,
    "line": Polyline,
    "point": Point,
    "circle": Polygon,
    "mask": Bitmap,
    "linestrip": Polyline,
}


def labelme_polygon_to_sly(coords: List[List[int]]) -> Polygon:
    coords = [[row, col] for (col, row) in coords]
    points = row_col_list_to_points(coords, do_round=True)
    return Polygon(points)


def labelme_rectangle_to_sly(coords: List[List[int]]) -> Rectangle:
    left, top = coords[0]
    right, bottom = coords[1]
    return Rectangle(int(top), int(left), int(bottom), int(right))


def labelme_line_to_sly(coords: List[List[int]]) -> Polyline:
    coords = [[row, col] for (col, row) in coords]
    points = row_col_list_to_points(coords, do_round=True)
    return Polyline(points)


def labelme_point_to_sly(coords: List[List[int]]) -> Point:
    col, row = coords[0]
    return Point(int(row), int(col))


def labelme_to_sly_bitmap(origin: List[List[int]], mask: str) -> Bitmap:
    origin = row_col_list_to_points([[row, col] for (col, row) in origin], do_round=True)
    img = Image.open(io.BytesIO(base64.b64decode(mask)))
    mask = np.array(img).astype(np.bool)
    return Bitmap(mask, origin=origin[0] if len(origin) > 0 else None)


def labelme_circle_to_sly(coords: List[List[int]]) -> Polygon:
    if len(coords) != 2:
        raise ValueError("Circle must have exactly 2 points: center and point on the circle.")
    if len(coords[0]) != 2 or len(coords[1]) != 2:
        raise ValueError("Each point must have exactly 2 coordinates: row and column.")
    center_col, center_row = int(coords[0][0]), int(coords[0][1])
    point_col, point_row = int(coords[1][0]), int(coords[1][1])
    radius = int(np.sqrt((center_row - point_row) ** 2 + (center_col - point_col) ** 2))
    points = []
    for i in range(0, 360, 10):
        col = int(center_col + radius * np.cos(np.radians(i)))
        row = int(center_row + radius * np.sin(np.radians(i)))
        points.append(PointLocation(row, col))
    return Polygon(points)


labelme_shape_types_to_convert_func = {
    "polygon": labelme_polygon_to_sly,
    "rectangle": labelme_rectangle_to_sly,
    "line": labelme_line_to_sly,
    "point": labelme_point_to_sly,
    "circle": labelme_circle_to_sly,
    "mask": labelme_to_sly_bitmap,
    "linestrip": labelme_line_to_sly,
}


def convert_labelme_to_sly(shape: Dict, obj_cls: ObjClass) -> Optional[Label]:
    try:
        shape_type = shape.get("shape_type")
        if shape_type is None:
            raise ValueError("Shape type is not specified.")
        convert_func = labelme_shape_types_to_convert_func.get(shape_type)
        if convert_func is None:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        coords = shape.get("points")
        if coords is None:
            raise ValueError("Shape coordinates are not specified. Skipping.")
        if shape_type == "mask":
            mask = shape.get("mask")
            geometry = convert_func(origin=coords, mask=mask)
        else:
            geometry = convert_func(coords)
        return Label(geometry, obj_cls)
    except Exception as e:
        logger.warn(f"Failed to convert shape: {shape_type}. Reason: {repr(e)}")
        return None


def decode_and_save_image_data(image_data: str, image_path: str) -> str:
    try:
        imencoded = zlib.decompress(base64.b64decode(image_data))
        n = np.frombuffer(imencoded, np.uint8)
        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    except zlib.error:
        # If the string is not compressed, we'll not use zlib.
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        imdecoded = np.array(img)

    imdecoded = cv2.cvtColor(imdecoded, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, imdecoded)
    return image_path


def get_image_from_data(ann_path: str, possible_image_path: Optional[str]) -> str:
    if possible_image_path is not None and file_exists(possible_image_path):
        return possible_image_path
    ann_json = load_json_file(ann_path)
    image_path = ann_json.get("imagePath")
    if image_path is None:
        image_path = Path(ann_path).with_suffix(".jpg").as_posix()
    if file_exists(image_path):
        return image_path

    # if contains encoded image data in base64
    image_data = ann_json.get("imageData")
    if image_data is not None:
        return decode_and_save_image_data(image_data, image_path)
    return


def generate_new_cls_name(
    meta: ProjectMeta,
    cls_name: str,
    expected_geometry_type: type,
    i: Optional[int] = None,
) -> str:
    new_cls_name = f"{cls_name}_{i}" if i else cls_name
    obj_cls: ObjClass = meta.get_obj_class(new_cls_name)
    if obj_cls is None:
        return new_cls_name
    elif obj_cls.geometry_type == expected_geometry_type:
        return new_cls_name
    return generate_new_cls_name(meta, cls_name, expected_geometry_type, i + 1 if i else 1)


def update_meta_from_labelme_annotation(meta: ProjectMeta, ann_path: str) -> ProjectMeta:
    """Update project meta from LabelMe annotation."""

    raw_json = load_json_file(ann_path)
    shapes = raw_json.get("shapes")
    json_updated = False
    for shape in shapes:
        cls_name = shape.get("label")
        shape_type = shape.get("shape_type")
        geometry_type = labelme_shape_types_to_sly_map.get(shape_type)
        if geometry_type is None:
            logger.warn(f"Unsupported shape type: {shape_type}. Please, contact support.")
        if not cls_name or not shape_type:
            continue
        obj_cls = meta.get_obj_class(cls_name)
        if obj_cls is None:
            obj_cls = ObjClass(cls_name, geometry_type)
            meta = meta.add_obj_class(obj_cls)
        elif obj_cls.geometry_type != geometry_type:
            new_cls_name = generate_new_cls_name(meta, cls_name, geometry_type)
            if new_cls_name != cls_name:
                logger.warn(
                    f"{ann_path}: shape type mismatch for class '{cls_name}'. Renamed to '{new_cls_name}'"
                )
                shape["label"] = new_cls_name
                json_updated = True
            if not meta.obj_classes.has_key(new_cls_name):
                obj_cls = ObjClass(new_cls_name, geometry_type)
                meta = meta.add_obj_class(obj_cls)
    if json_updated:
        dump_json_file(raw_json, ann_path)

    return meta


def create_supervisely_annotation(
    item,
    project_meta: ProjectMeta,
    renamed_classes: Dict[str, str],
) -> Annotation:
    ann = Annotation.from_img_path(item.path)
    h, w = ann.img_size
    if item.ann_data is None:
        return ann
    raw_json = load_json_file(item.ann_data)
    if raw_json.get("imageHeight") != h or raw_json.get("imageWidth") != w:
        logger.warn("Image size in annotation does not match the actual image size. Skipping.")
        return ann

    shapes = raw_json.get("shapes")
    labels = []
    for shape in shapes:
        cls_name = shape.get("label")
        cls_name = renamed_classes.get(cls_name, cls_name)
        obj_class = project_meta.get_obj_class(cls_name)
        if obj_class is None:
            logger.warn(f"Object class '{cls_name}' not found in project meta. Skipping.")
            continue

        label = convert_labelme_to_sly(shape, obj_class)
        if label is not None:
            labels.append(label)
    labels = validate_image_bounds(labels, Rectangle.from_size(ann.img_size))
    ann = ann.add_labels(labels)

    return ann
