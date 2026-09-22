import os
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

import supervisely as sly
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


def download_init_mask(api: sly.Api, figure_id, image_id) -> sly.Bitmap:
    """Download the init mask of a figure by downloading the whole image annotation.

    .. deprecated::
        Resolving the init mask from ``figure_id``/``init_figure`` is deprecated.
        Senders should put the mask into the request context instead (see
        :func:`get_init_mask_from_context`), which needs no annotation download.
        This path only works for bitmap figures: a polygon, multipolygon or
        AnyShape figure fails in :meth:`supervisely.Bitmap.from_json` here.
    """
    ann_json = api.annotation.download_json(image_id)
    labels = [label for label in ann_json["objects"] if label["id"] == figure_id]
    assert len(labels) > 0, f"Label with id {figure_id} not found in image {image_id}."
    label = labels[0]
    bitmap = sly.Bitmap.from_json(label)
    return bitmap


def bitmap_to_mask(bitmap: sly.Bitmap, h, w):
    mask = np.zeros((h, w), bool)
    bitmap.to_bbox().get_cropped_numpy_slice(mask)[:] = bitmap.data
    mask = (mask * 255).astype(np.uint8)
    return mask


class InitMaskDecodeError(ValueError):
    """Request context carries an init ``mask`` that cannot be decoded."""


def get_init_mask_from_context(context: dict) -> Optional[sly.Bitmap]:
    """Build the init mask bitmap from the ``mask`` field of the request context.

    The field is optional and has the form
    ``{"data": <base64 string>, "origin": {"x": <int>, "y": <int>}}``, where
    ``data`` is the same encoding :class:`supervisely.Bitmap` uses on the wire
    (base64 of a zlib-compressed PNG, non-zero pixels are foreground) and
    ``origin`` is the top-left corner of the mask in full image coordinates.
    When it is present it fully replaces the deprecated ``figure_id`` lookup, so
    no annotation is downloaded and figures of any geometry can be sent back.

    :param context: Request context of a smart tool request.
    :type context: dict
    :returns: Bitmap built from the context, or None when ``mask`` is absent or null.
    :rtype: :class:`supervisely.Bitmap` or None
    :raises InitMaskDecodeError: if ``mask`` carries a value that cannot be decoded.
    """
    mask = context.get("mask")
    if mask is None:
        return None
    try:
        origin = mask["origin"]
        data = sly.Bitmap.base64_2_data(mask["data"])
        return sly.Bitmap(
            data=data,
            origin=sly.PointLocation(row=int(origin["y"]), col=int(origin["x"])),
        )
    except Exception as exc:
        raise InitMaskDecodeError(f"Can not decode the init mask from request: {exc}") from exc


def bitmap_to_mask_in_crop(bitmap: sly.Bitmap, crop) -> np.ndarray:
    """Rasterize a bitmap into the crop region without knowing the image size.

    Returns exactly what ``crop_image(crop, bitmap_to_mask(bitmap, h, w))``
    returns for any image that contains both the crop and the bitmap.
    """
    bbox = bitmap.to_bbox()
    h = max(int(crop[1]["y"]) + 1, bbox.bottom + 1)
    w = max(int(crop[1]["x"]) + 1, bbox.right + 1)
    return crop_image(crop, bitmap_to_mask(bitmap, h, w))
