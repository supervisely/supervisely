import os
from copy import deepcopy
from typing import Callable, MutableMapping, Optional, Tuple

import numpy as np

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from supervisely.io.fs import silent_remove
from supervisely.sly_logger import logger


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


# Key of the inline source figure in a Smart Tool request context, and of its two fields.
# The Smart Tool sends the figure it is editing along with the clicks; `figure_id` +
# `init_figure` is the legacy way of saying the same thing, and it made the app read the
# annotation back from the API, which only ever worked for bitmap figures.
MASK = "mask"
GEOMETRY_TYPE = "geometry_type"
GEOMETRY = "geometry"


def mask_geometry_from_json(mask_json: dict) -> Geometry:
    """Build the geometry a Smart Tool request carries in its ``mask`` field.

    :param mask_json: ``{"geometry_type": <Supervisely geometry name>, "geometry": <geometry json>}``.
        A bare geometry json is read as a bitmap, which is what the older clients that
        send a ``geometry`` field mean by it.
    :type mask_json: dict
    :returns: geometry of the figure the Smart Tool is editing
    :rtype: :class:`~supervisely.geometry.geometry.Geometry`
    """
    geometry_json = mask_json.get(GEOMETRY, mask_json)
    geometry_type = mask_json.get(GEOMETRY_TYPE, sly.Bitmap.geometry_name())
    return sly.deserialize_geometry(geometry_type, geometry_json)


def geometry_to_mask(geometry: Geometry, image_size: Tuple[int, int]) -> np.ndarray:
    """Rasterize a geometry into a full-image mask, 0 outside and 255 inside.

    :param geometry: any geometry that can be drawn - bitmap, polygon, multipolygon, ...
    :type geometry: :class:`~supervisely.geometry.geometry.Geometry`
    :param image_size: (height, width) of the image the geometry belongs to
    :type image_size: Tuple[int, int]
    :returns: mask of the same size as the image
    :rtype: np.ndarray
    """
    mask = np.zeros(image_size, np.uint8)
    geometry.draw(mask, color=255)
    return mask


def get_smart_tool_init_mask(
    context: dict,
    image_size: Tuple[int, int],
    api: Optional[sly.Api] = None,
    cache: Optional[MutableMapping] = None,
) -> Optional[np.ndarray]:
    """Build the init mask for one Smart Tool request, in the size of the full image.

    This is what lets a model refine a figure that already exists. The figure arrives
    inline in ``context["mask"]`` and may be of any drawable shape, so a polygon or a
    multipolygon can be handed back to the model just like a bitmap.

    The Smart Tool only sends the figure with the first request of a session, so pass a
    ``cache`` to keep it for the clicks that follow.

    :param context: Smart Tool request context
    :type context: dict
    :param image_size: (height, width) of the image being annotated
    :type image_size: Tuple[int, int]
    :param api: API used by the deprecated ``figure_id`` path only
    :type api: :class:`~supervisely.api.api.Api`, optional
    :param cache: mutable mapping keeping the session's geometry between clicks
    :type cache: MutableMapping, optional
    :returns: mask of the figure being edited, or None when the request starts from scratch
    :rtype: np.ndarray, optional

    :Usage example:

     .. code-block:: python

        init_mask = functional.get_smart_tool_init_mask(
            smtool_state, image_np.shape[:2], api=api, cache=self._init_mask_cache
        )
        settings["init_mask"] = init_mask
    """
    cache_key = context.get("figure_id") or context.get("local_figure_id")
    geometry = None

    mask_json = context.get(MASK) or context.get(GEOMETRY)
    if mask_json is not None:
        try:
            geometry = mask_geometry_from_json(mask_json)
        except Exception:
            # A figure the app cannot read is not worth failing the click over: the model
            # still segments from the clicks alone, it just starts without a prior.
            logger.warning("Smart Tool init mask cannot be read, ignoring it.", exc_info=True)
            return None
    elif context.get("init_figure") is True and api is not None and context.get("image_id"):
        logger.warning(
            "Smart Tool init mask is requested by figure_id, which is deprecated and "
            "supports bitmap figures only. Send the figure in the 'mask' field instead."
        )
        geometry = download_init_mask(api, context.get("figure_id"), context["image_id"])
    elif cache is not None and cache_key is not None:
        geometry = cache.get(cache_key)

    if geometry is None:
        return None

    if cache is not None and cache_key is not None:
        cache[cache_key] = geometry

    try:
        return geometry_to_mask(geometry, image_size)
    except Exception:
        # Same reasoning as above: a geometry that does not fit this image (a stale
        # figure, a wrong entity) must not take the whole click down.
        logger.warning("Smart Tool init mask cannot be rasterized, ignoring it.", exc_info=True)
        return None
