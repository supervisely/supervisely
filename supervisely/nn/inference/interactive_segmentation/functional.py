import os
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np

import supervisely as sly
from supervisely.annotation.annotation import AnnotationJsonFields
from supervisely.annotation.label import LabelJsonFields
from supervisely.geometry.constants import (
    EXTERIOR,
    GEOMETRY_SHAPE,
    GEOMETRY_TYPE,
    INTERIOR,
    PARTS,
    POINTS,
)
from supervisely.io.fs import silent_remove


def get_image_by_hash(hash, save_path, api: sly.Api):
    api.image.download_paths_by_hashes([hash], [save_path])
    base_image = sly.image.read(save_path)
    silent_remove(save_path)
    return base_image


def download_image_from_context(
    context: dict,
    api: sly.Api,
    output_dir: str,
    cache_load_img: Callable[[sly.Api, int], np.ndarray] = None,
    cache_load_frame: Callable[[sly.Api, int, int], np.ndarray] = None,
    cache_load_img_hash: Callable[[sly.Api, str], np.ndarray] = None,
):
    if "image_id" in context:
        if cache_load_img is not None:
            return cache_load_img(api, context["image_id"])
        return api.image.download_np(context["image_id"])
    elif "image_hash" in context:
        if cache_load_img_hash is not None:
            return cache_load_img_hash(api, context["image_hash"])
        img_path = os.path.join(output_dir, "base_image.png")
        return get_image_by_hash(context["image_hash"], img_path, api=api)
    elif "volume" in context:
        volume_id = context["volume"]["volume_id"]
        slice_index = context["volume"]["slice_index"]
        normal = context["volume"]["normal"]
        window_center = context["volume"]["window_center"]
        window_width = context["volume"]["window_width"]
        plane = sly.Plane.get_name(normal)
        return api.volume.download_slice_np(
            volume_id, slice_index, plane, window_center, window_width
        )
    elif "video" in context:
        if cache_load_frame is not None:
            return cache_load_frame(
                api,
                context["video"]["video_id"],
                context["video"]["frame_index"],
            )
        return api.video.frame.download_np(
            context["video"]["video_id"], context["video"]["frame_index"]
        )
    elif "pcd_related_image_id" in context:
        if cache_load_img is not None:
            return cache_load_img(api, context["pcd_related_image_id"], related=True)
        return api.pointcloud.download_related_image(context["pcd_related_image_id"])
    else:
        raise Exception("Project type is not supported")


def crop_image(crop, image_np):
    x1, y1 = crop[0]["x"], crop[0]["y"]
    x2, y2 = crop[1]["x"], crop[1]["y"]
    bbox = sly.Rectangle(y1, x1, y2, x2)
    img_crop = sly.image.crop(image_np, bbox)
    return img_crop


def transform_clicks_to_crop(crop, clicks: list):
    clicks = deepcopy(clicks)
    for click in clicks:
        click["x"] -= crop[0]["x"]
        click["y"] -= crop[0]["y"]
    return clicks


def validate_click_bounds(crop, clicks: list):
    x_max = crop[1]["x"] - crop[0]["x"]  # width
    y_max = crop[1]["y"] - crop[0]["y"]  # height
    for click in clicks:
        is_in_bbox = (
            click["x"] >= 0 and click["y"] >= 0 and click["x"] <= x_max and click["y"] <= y_max
        )
        if not is_in_bbox:
            return False
    return True


def format_bitmap(bitmap: sly.Bitmap, crop):
    bitmap_json = bitmap.to_json()["bitmap"]
    bitmap_origin = bitmap_json["origin"]
    bitmap_origin = {
        "x": crop[0]["x"] + bitmap_origin[0],
        "y": crop[0]["y"] + bitmap_origin[1],
    }
    bitmap_data = bitmap_json["data"]
    return bitmap_origin, bitmap_data


def get_hash_from_context(context: dict):
    if "image_id" in context:
        return str(context["image_id"])
    elif "image_hash" in context:
        return context["image_hash"]
    elif "volume" in context:
        volume_id = context["volume"]["volume_id"]
        slice_index = context["volume"]["slice_index"]
        normal = context["volume"]["normal"]
        window_center = context["volume"]["window_center"]
        window_width = context["volume"]["window_width"]
        plane = sly.Plane.get_name(normal)
        return "_".join(map(str, [volume_id, slice_index, plane, window_center, window_width]))
    elif "video" in context:
        return "_".join(map(str, [context["video"]["video_id"], context["video"]["frame_index"]]))
    elif "pcd_related_image_id" in context:
        return str(context["pcd_related_image_id"])
    else:
        raise Exception("Project type is not supported")


class InitMaskError(RuntimeError):
    """Raised when the initial Smart Tool figure cannot be normalized into a mask."""


#: Concrete label geometries that can be used as a Smart Tool initial figure. A label of an
#: AnyShape class stores its concrete type in the very same field, so it needs no special case.
SUPPORTED_INIT_GEOMETRIES = (
    sly.Bitmap.geometry_name(),
    sly.Polygon.geometry_name(),
    sly.Multipolygon.geometry_name(),
)


def get_init_geometry_type(label: dict) -> str:
    """Concrete geometry type of a downloaded annotation label.

    :param label: Label in Supervisely json format.
    :type label: dict
    :returns: Geometry type name, e.g. "bitmap", "polygon" or "multipolygon".
    :rtype: str
    :raises InitMaskError: if the type is neither stored nor unambiguously recoverable.
    """
    geometry_type = label.get(GEOMETRY_TYPE) or label.get(GEOMETRY_SHAPE)
    if geometry_type is not None and geometry_type != sly.AnyGeometry.geometry_name():
        return geometry_type
    # Legacy payloads may omit the type. Only unambiguous keys may be used to restore it:
    # "points" is shared by polygon, rectangle, polyline and point, so it is not one of them.
    if sly.Bitmap.geometry_name() in label:
        return sly.Bitmap.geometry_name()
    if PARTS in label:
        return sly.Multipolygon.geometry_name()
    raise InitMaskError("Geometry type of the initial figure is missing and cannot be inferred.")


def _polygon_parts_json(label: dict) -> dict:
    """Repacks a polygon label as a single-part multipolygon.

    Both geometries are made of the same exterior/interior rings, so reading them through
    one parser keeps the ring validation and the hole handling identical. A missing
    (i.e. empty) "interior" field is tolerated.
    """
    points = label.get(POINTS)
    if not isinstance(points, dict):
        raise InitMaskError(f'"{POINTS}" field is required to read a polygon initial figure.')
    return {PARTS: [{EXTERIOR: points.get(EXTERIOR), INTERIOR: points.get(INTERIOR, [])}]}


def _mask_to_bitmap(mask: np.ndarray, drow: int = 0, dcol: int = 0) -> sly.Bitmap:
    """Builds a Bitmap from a rasterized mask, placing its origin at the mask tight bbox."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        raise InitMaskError("Initial figure is empty after rasterization within the image bounds.")
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return sly.Bitmap(
        mask[top : bottom + 1, left : right + 1],
        origin=sly.PointLocation(row=int(top) + drow, col=int(left) + dcol),
    )


def _clip_bitmap(bitmap: sly.Bitmap, img_size: Tuple[int, int]) -> sly.Bitmap:
    """Clips a Bitmap to the image bounds, keeping its origin in image coordinates."""
    h, w = img_size
    data = bitmap.data
    top, left = bitmap.origin.row, bitmap.origin.col
    bottom, right = top + data.shape[0], left + data.shape[1]
    if top >= 0 and left >= 0 and bottom <= h and right <= w:
        return bitmap
    new_top, new_left = max(top, 0), max(left, 0)
    new_bottom, new_right = min(bottom, h), min(right, w)
    if new_bottom <= new_top or new_right <= new_left:
        raise InitMaskError("Initial figure lies completely outside of the image bounds.")
    clipped = data[new_top - top : new_bottom - top, new_left - left : new_right - left]
    if not clipped.any():
        raise InitMaskError("Initial figure is empty after clipping to the image bounds.")
    return sly.Bitmap(clipped, origin=sly.PointLocation(row=new_top, col=new_left))


def _rasterize_geometry(
    geometry: sly.Multipolygon, img_size: Optional[Tuple[int, int]]
) -> sly.Bitmap:
    """Rasterizes a vector geometry into a Bitmap positioned in image coordinates.

    Every polygon part is filled with its own holes cut out and then unioned with the parts
    already drawn, so a hole of one part never erases another one. Points outside of the
    canvas are clipped by the drawing primitives.
    """
    drow, dcol = 0, 0
    if img_size is not None:
        h, w = img_size
    else:
        # Image size is unknown: rasterize on the geometry bbox instead, without clipping.
        bbox = geometry.to_bbox()
        h, w = bbox.height, bbox.width
        drow, dcol = bbox.top, bbox.left
        geometry = geometry.translate(-drow, -dcol)
    if h <= 0 or w <= 0:
        raise InitMaskError(f"Can not rasterize the initial figure on a canvas of size {(h, w)}.")
    canvas = np.zeros((h, w), np.uint8)
    geometry.draw(canvas, color=1)
    return _mask_to_bitmap(canvas.astype(bool), drow, dcol)


def label_to_init_bitmap(label: dict, img_size: Optional[Tuple[int, int]] = None) -> sly.Bitmap:
    """Normalizes a downloaded annotation label into a Bitmap in image coordinates.

    Bitmap labels are returned as is (only clipped to the image), polygon and multipolygon
    labels are rasterized. Labels of an AnyShape class are dispatched by their concrete
    geometry type in the same way.

    :param label: Label in Supervisely json format.
    :type label: dict
    :param img_size: Image size (height, width) used to clip the figure, if known.
    :type img_size: Tuple[int, int], optional
    :returns: Initial mask as a Bitmap.
    :rtype: :class:`~supervisely.geometry.bitmap.Bitmap`
    :raises InitMaskError: if the geometry is unsupported, malformed or empty.
    """
    geometry_type = get_init_geometry_type(label)
    try:
        if geometry_type == sly.Bitmap.geometry_name():
            bitmap = sly.Bitmap.from_json(label)
            return bitmap if img_size is None else _clip_bitmap(bitmap, img_size)
        if geometry_type == sly.Polygon.geometry_name():
            parts_json = _polygon_parts_json(label)
        elif geometry_type == sly.Multipolygon.geometry_name():
            parts_json = label
        else:
            raise InitMaskError(
                f"Geometry '{geometry_type}' is not supported as a Smart Tool initial figure. "
                f"Supported geometries: {', '.join(SUPPORTED_INIT_GEOMETRIES)}."
            )
        return _rasterize_geometry(sly.Multipolygon.from_json(parts_json), img_size)
    except InitMaskError:
        raise
    except Exception as exc:
        raise InitMaskError(
            f"Failed to read the '{geometry_type}' initial figure: {exc}"
        ) from exc


def _get_ann_image_size(ann_json: dict) -> Optional[Tuple[int, int]]:
    """Image size (height, width) stored in a downloaded annotation, if it is usable."""
    size = ann_json.get(AnnotationJsonFields.IMG_SIZE) or {}
    height = size.get(AnnotationJsonFields.IMG_SIZE_HEIGHT)
    width = size.get(AnnotationJsonFields.IMG_SIZE_WIDTH)
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


def download_init_mask(api: sly.Api, figure_id, image_id) -> sly.Bitmap:
    """Downloads the initial Smart Tool figure and normalizes it into a Bitmap.

    :param api: Supervisely API.
    :type api: :class:`~supervisely.api.api.Api`
    :param figure_id: ID of the label to be used as the initial figure.
    :type figure_id: int
    :param image_id: ID of the image the label belongs to.
    :type image_id: int
    :returns: Initial mask as a Bitmap in image coordinates.
    :rtype: :class:`~supervisely.geometry.bitmap.Bitmap`
    :raises InitMaskError: if the figure is missing, unsupported or malformed.
    """
    if figure_id is None:
        raise InitMaskError(f"Id of the initial figure is not provided for image {image_id}.")
    ann_json = api.annotation.download_json(image_id)
    labels = [
        label
        for label in ann_json.get(AnnotationJsonFields.LABELS, [])
        if label.get(LabelJsonFields.ID) == figure_id
    ]
    if len(labels) == 0:
        raise InitMaskError(f"Label with id {figure_id} not found in image {image_id}.")
    return label_to_init_bitmap(labels[0], _get_ann_image_size(ann_json))


def bitmap_to_mask(bitmap: sly.Bitmap, h, w):
    """Places a Bitmap on a full-image mask, clipping the parts outside of the image."""
    mask = np.zeros((h, w), bool)
    data = bitmap.data.astype(bool)
    top, left = bitmap.origin.row, bitmap.origin.col
    bottom, right = top + data.shape[0], left + data.shape[1]
    new_top, new_left = max(top, 0), max(left, 0)
    new_bottom, new_right = min(bottom, h), min(right, w)
    if new_bottom > new_top and new_right > new_left:
        mask[new_top:new_bottom, new_left:new_right] = data[
            new_top - top : new_bottom - top, new_left - left : new_right - left
        ]
    mask = (mask * 255).astype(np.uint8)
    return mask
