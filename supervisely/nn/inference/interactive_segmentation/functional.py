import os
from copy import deepcopy
from typing import Callable, Tuple

import numpy as np

import supervisely as sly
from supervisely.annotation.annotation import AnnotationJsonFields
from supervisely.annotation.label import LabelJsonFields
from supervisely.geometry.constants import DATA, ORIGIN
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
    """Raised when the initial Smart Tool mask cannot be read from a request."""


def _origin_coordinate(value, name: str) -> int:
    """Reads one integer coordinate of the initial mask origin."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InitMaskError(
            f'"{ORIGIN}" of the initial mask must be a pair of integers, got {name}={value!r}.'
        )
    if isinstance(value, float) and not value.is_integer():
        raise InitMaskError(
            f'"{ORIGIN}" of the initial mask must be a pair of integers, got {name}={value!r}.'
        )
    return int(value)


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
        raise InitMaskError("Initial mask lies completely outside of the image bounds.")
    clipped = data[new_top - top : new_bottom - top, new_left - left : new_right - left]
    if not clipped.any():
        raise InitMaskError("Initial mask is empty after clipping to the image bounds.")
    return sly.Bitmap(clipped, origin=sly.PointLocation(row=new_top, col=new_left))


def decode_init_mask(mask_json: dict, img_size: Tuple[int, int]) -> sly.Bitmap:
    """Reads the initial Smart Tool mask supplied by the client into a positioned Bitmap.

    The mask is a tight binary bitmap in image (or video frame) coordinates:
    ``{"origin": [x, y], "data": <base64 encoded bitmap payload>}``, where the payload uses
    the very same encoding as the "bitmap" field of the Supervisely annotation format. The
    figure is placed at its origin and clipped to the image, so the caller never has to
    download the annotation of the edited figure.

    :param mask_json: Initial mask as sent in the request context.
    :type mask_json: dict
    :param img_size: Image size (height, width) the mask is placed on.
    :type img_size: Tuple[int, int]
    :returns: Initial mask as a Bitmap in image coordinates.
    :rtype: :class:`~supervisely.geometry.bitmap.Bitmap`
    :raises InitMaskError: if the payload is malformed, empty or fully outside of the image.
    """
    if not isinstance(mask_json, dict):
        raise InitMaskError(
            f'Initial mask must be an object with "{ORIGIN}" and "{DATA}" fields, '
            f"got {type(mask_json).__name__}."
        )
    origin = mask_json.get(ORIGIN)
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise InitMaskError(f'"{ORIGIN}" of the initial mask must be an [x, y] pair.')
    col = _origin_coordinate(origin[0], "x")
    row = _origin_coordinate(origin[1], "y")

    data = mask_json.get(DATA)
    if not isinstance(data, str) or not data.strip():
        raise InitMaskError(f'"{DATA}" of the initial mask must be a non-empty encoded string.')
    try:
        mask = sly.Bitmap.base64_2_data(data)
    except Exception as exc:
        raise InitMaskError(f"Failed to decode the initial mask data: {exc}") from exc
    if mask.ndim != 2:
        raise InitMaskError(
            f"Initial mask must decode into a 2-dimensional mask, got {mask.ndim} dimensions."
        )
    if not mask.any():
        raise InitMaskError("Initial mask is empty.")

    bitmap = sly.Bitmap(mask, origin=sly.PointLocation(row=row, col=col))
    return _clip_bitmap(bitmap, img_size)


def download_init_mask(api: sly.Api, figure_id, image_id) -> sly.Bitmap:
    """Downloads the initial Smart Tool figure by its id.

    .. deprecated::
        Clients must send the initial figure as a ``mask`` in the request context and let
        :func:`decode_init_mask` read it. This download is kept only for the callers that
        still identify the initial figure by ``figure_id``, and it requires the figure to be
        an already saved bitmap.

    :param api: Supervisely API.
    :type api: :class:`~supervisely.api.api.Api`
    :param figure_id: ID of the label to be used as the initial figure.
    :type figure_id: int
    :param image_id: ID of the image the label belongs to.
    :type image_id: int
    :returns: Initial mask as a Bitmap in image coordinates.
    :rtype: :class:`~supervisely.geometry.bitmap.Bitmap`
    :raises InitMaskError: if the figure id is missing, unknown or does not hold a bitmap.
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
    try:
        return sly.Bitmap.from_json(labels[0])
    except Exception as exc:
        raise InitMaskError(
            f"Figure {figure_id} of image {image_id} can not be read as an initial bitmap: {exc}"
        ) from exc


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
